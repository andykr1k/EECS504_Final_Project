"""
Generates the dataset from the preprocessed videos.
Uses SAM to segment people in the videos and track them across clips.

#### *Dataset Structure:*
```
<processed_video_stem>_segmentations/
  <video_slice_stem>/
    <video_slice_stem>_<frame_idx>_frame.png
    <video_slice_stem>_<frame_idx>_segmentation.npz
    <video_slice_stem>_<frame_idx>_sidecar.json
```
These folders can live anywhere in the file tree so when loading the dataset we need to search for directories with this structure.

#### *Segmentations Structure:*
Numpy array of same shape as image, stored in the .npz archive under the key 'segmentation'. Values are the person ids. -1 for background.

#### *Sidecar Structure:*
```
{
  "video_path": str,
  "frame_idx": int,
  "visible_person_ids": list[int], # List of person ids visible in this frame
}
```
"""

try:
    import torch
    import torchvision  # We don't use this, but we need to check it because SMA3 import fails with a weird error if it is not installed
    import accelerate
except ImportError:
    print("Please install torch and torchvision")
    exit(1)

from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import cv2
import json
import yaml

from dotenv import load_dotenv
load_dotenv()

class Sam3VideoPersonSegmenter:
    def __init__(self, device: str | None = None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self.model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
        self.processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    def segment_video(self, video_path: Path | str, prompt: str = "person", top_n: int = 10, 
                      min_confidence: float = 0.75, min_frame_confidence: float = 0.5,
                      min_mask_ratio: float = 0.007, min_frames_visible: int = 10, min_frame_mask_ratio: float = 0.005,
                      max_num_frames: int | None = None, skip_frames: int = 1):
        video_path = Path(video_path)

        video_frames, _ = load_video(str(video_path))

        frame_indices = list(range(0, len(video_frames), skip_frames))
        video_frames = [video_frames[i] for i in frame_indices]

        inference_session = self.processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            # processing_device="cpu",
            # video_storage_device="cpu",
            dtype=torch.bfloat16,
        )

        inference_session = self.processor.add_text_prompt(
            inference_session=inference_session,
            text=prompt,
        )

        outputs_per_frame = {}
        # Pass show_progress_bar=True to display a tqdm progress bar.
        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=inference_session, max_frame_num_to_track=max_num_frames,
            show_progress_bar=True,
        ):
            processed_outputs = self.processor.postprocess_outputs(inference_session, model_outputs)
            outputs_per_frame[model_outputs.frame_idx] = processed_outputs

        # Convert to a more easily usable format where each key is the object id and the value contains information about detections across all frames
        object_data = {}
        total_pixels = video_frames[0].shape[0] * video_frames[0].shape[1]
        for i, frame_outputs in outputs_per_frame.items():
            frame_idx = frame_indices[i]
            frame_obj_ids = frame_outputs["object_ids"]
            frame_scores = frame_outputs["scores"]
            frame_boxes = frame_outputs["boxes"]
            frame_masks = frame_outputs["masks"]
            for j in range(len(frame_obj_ids)):
                score = float(frame_scores[j])
                if score < min_frame_confidence:
                    continue

                mask_ratio = float(frame_masks[j].sum()) / total_pixels
                if mask_ratio < min_frame_mask_ratio:
                    continue

                obj_id = int(frame_obj_ids[j])
                if obj_id not in object_data:
                    object_data[obj_id] = []
                
                object_frame_data = {
                    "frame_idx": int(frame_idx),
                    "score": score,
                    "box": [int(x) for x in frame_boxes[j]],
                    "mask": frame_masks[j].cpu().numpy(),
                }
                object_data[obj_id].append(object_frame_data)

        total_frames = len(outputs_per_frame)
        obj_summary_data = {}
        for obj_id, obj_data in object_data.items():
            # Collect proportion of frames visible, average score, average mask ratio
            visible_frames = len(obj_data)
            if visible_frames == 0:
                del object_data[obj_id]
                continue

            proportion_visible = visible_frames / total_frames
            avg_score = sum([frame_data["score"] for frame_data in obj_data]) / visible_frames
            avg_mask_ratio = sum([frame_data["mask"].sum() for frame_data in obj_data]) / (visible_frames * total_pixels)
            
            obj_summary_data[obj_id] = {
                "frames_visible": visible_frames,
                "proportion_visible": proportion_visible,
                "avg_score": avg_score,
                "avg_mask_ratio": avg_mask_ratio.item(),
            }

        # Filter out objects that do not meet the criteria
        filtered_obj_ids = []
        for obj_id in obj_summary_data:
            meets_criteria = True
            if obj_summary_data[obj_id]["frames_visible"] < min_frames_visible:
                meets_criteria = False
            if obj_summary_data[obj_id]["avg_score"] < min_confidence:
                meets_criteria = False
            if obj_summary_data[obj_id]["avg_mask_ratio"] < min_mask_ratio:
                meets_criteria = False
            if meets_criteria:
                filtered_obj_ids.append(obj_id)
        # Find the ids of the top n objects based on the average mask ratio
        filtered_obj_ids = sorted(filtered_obj_ids, key=lambda x: obj_summary_data[x]["avg_mask_ratio"], reverse=True)[:top_n]
        
        filtered_obj_data = {obj_id: obj_data for obj_id, obj_data in object_data.items() if obj_id in filtered_obj_ids}
        filtered_obj_summary_data = {obj_id: obj_summary_data[obj_id] for obj_id in filtered_obj_ids}

        return filtered_obj_data, filtered_obj_summary_data, video_frames, frame_indices

    def render_masked_video(self, video_frames: list, obj_data: dict, output_path: Path | str, fps: float = 30.0):
        filtered_obj_ids = obj_data.keys()
        np.random.seed(42) # For consistent colors across runs
        colors = {obj_id: np.random.randint(0, 255, (3,), dtype=np.uint8).tolist() for obj_id in filtered_obj_ids}

        used_frames = set()
        for obj_id, obj_frames in obj_data.items():
            for frame_data in obj_frames:
                used_frames.add(frame_data["frame_idx"])

        # 2. Re-index filtered masks by frame index for faster rendering
        masks_by_frame = {i: [] for i in used_frames}
        for obj_id, obj_frames in obj_data.items():
            for frame_data in obj_frames:
                frame_idx = frame_data["frame_idx"]
                mask = frame_data["mask"]
                masks_by_frame[frame_idx].append((obj_id, mask))

        # 3. Initialize OpenCV VideoWriter
        # Determine width and height from the first frame
        first_frame = np.array(video_frames[0])
        height, width = first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        print(f"Writing segmented video to {output_path} at {fps} fps...")

        # 4. Render frames
        for i, frame in enumerate(video_frames):
            if i not in used_frames:
                continue
            # Convert RGB (from transformers) to BGR (for OpenCV)
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            overlay = frame_bgr.copy()
            
            # Apply all masks for this frame
            for obj_id, mask in masks_by_frame.get(i, []):
                color = colors[obj_id]
                
                # Convert mask to numpy boolean array if it's a tensor
                if torch.is_tensor(mask):
                    mask_np = mask.cpu().numpy().astype(bool)
                else:
                    mask_np = np.array(mask).astype(bool)
                
                # Squeeze in case the mask has a channel dimension like (1, H, W)
                if mask_np.ndim == 3:
                    mask_np = mask_np.squeeze()

                # Color the mask area on the overlay
                overlay[mask_np] = color
            
            # Blend the overlay with the original frame (alpha = 0.5)
            alpha = 0.5
            frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
            
            # Write to video
            out.write(frame_bgr)

        out.release()
        print("Video rendering complete.")


@dataclass
class Config:
    enabled: bool
    sam_prompt: str
    sam_skip_frames: int
    save_skip_frames: int
    max_scene_len_frames: int

def segment_video_slice(config: Config, video_segmenter: Sam3VideoPersonSegmenter, video_slice_path: Path | str, out_dir: Path | str):
    print(f"Segmenting video slice {video_slice_path}...")
    video_slice_path = Path(video_slice_path)
    out_dir = Path(out_dir)

    obj_data, obj_summary_data, video_frames, frame_indices = video_segmenter.segment_video(
        video_slice_path,
        prompt=config.sam_prompt,
        skip_frames=config.sam_skip_frames,
        top_n=1000
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    height, width = video_frames[0].shape[:2]

    all_video_frames, _ = load_video(str(video_slice_path))
    video_segmenter.render_masked_video(all_video_frames, obj_data, out_dir / f"{video_slice_path.stem}_segmented.mp4")

    frame_sidecars: dict[int, dict] = {}
    frame_segmentations: dict[int, np.ndarray] = {}
    for frame_idx in frame_indices:
        sidecar_data = {
            "video_path": str(video_slice_path),
            "frame_idx": frame_idx,
            "visible_person_ids": []
        }
        frame_sidecars[frame_idx] = sidecar_data
        frame_segmentations[frame_idx] = np.full((height, width), -1, dtype=np.int32)

    for obj_id, obj_datum in obj_data.items():
        for frame_data in obj_datum:
            frame_idx = frame_data["frame_idx"]
            frame_sidecars[frame_idx]["visible_person_ids"].append(obj_id)

            mask = frame_data["mask"]
            where_mask = np.where(mask)
            frame_segmentations[frame_idx][where_mask] = obj_id

    last_saved_frame_idx = -float('inf')
    for i in range(len(frame_indices)):
        frame_idx = frame_indices[i]
        frame = video_frames[i]
        
        segmentation = frame_segmentations[frame_idx]
        sidecar = frame_sidecars[frame_idx]

        if len(sidecar["visible_person_ids"]) == 0:
            print(f"No people detected in frame {frame_idx}")
            continue

        # Skip if it's too close to the last saved frame
        if frame_idx - last_saved_frame_idx < config.save_skip_frames:
            continue

        frame_path = out_dir / f"{video_slice_path.stem}_{frame_idx}_frame.png"
        sidecar_path = out_dir / f"{video_slice_path.stem}_{frame_idx}_sidecar.json"
        segmentation_path = out_dir / f"{video_slice_path.stem}_{frame_idx}_segmentation.npz"

        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)
        np.savez_compressed(str(segmentation_path), segmentation=segmentation)

        last_saved_frame_idx = frame_idx

def segment_preprocessed_video(config: Config, video_segmenter: Sam3VideoPersonSegmenter, processed_video_path: Path | str, out_dir: Path | str):
    print(f"Segmenting preprocessed video {processed_video_path}...")
    processed_video_path = Path(processed_video_path)
    out_dir = Path(out_dir)

    video_slices = processed_video_path.glob("*.mp4")
    for video_slice_path in video_slices:
        out_imgs_dir = out_dir / video_slice_path.stem
        if out_imgs_dir.exists():
            print(f"Skipping {video_slice_path} (already processed)")
            continue

        segment_video_slice(config, video_segmenter, video_slice_path, out_imgs_dir)

def main(parent_dir: Path | str):
    parent_dir = Path(parent_dir)
    video_segmenter = Sam3VideoPersonSegmenter()
    for dir_path in parent_dir.glob("*"):
        if not dir_path.is_dir():
            continue

        config_path = dir_path / "config.yaml"
        assert config_path.exists(), f"Config not found at {config_path}"
        config = Config(**yaml.safe_load(config_path.read_text()))
        print(f"Config: {config}")

        if not config.enabled:
            print(f"Skipping {dir_path} (disabled)")
            continue

        for processed_video_path in dir_path.glob("*_processed"):
            if not processed_video_path.is_dir():
                continue
            out_dir = dir_path / f"{processed_video_path.stem}_segmentations"
            segment_preprocessed_video(config, video_segmenter, processed_video_path, out_dir)
    

if __name__ == "__main__":
    main("/z/dat/person_reid/internal/input_videos")
