try:
    import torch
    import torchvision  # We don't use this, but we need to check it because SMA3 import fails with a weird error if it is not installed
    import accelerate
except ImportError:
    print("Please install torch and torchvision")
    exit(1)

from transformers import Sam3Processor, Sam3Model
from PIL import Image
import requests

from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from transformers.video_utils import load_video

from accelerate import Accelerator
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import cv2

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
        # self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        # self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
        self.processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    # def segment(self, video_path: Path | str, prompt: str = "person", top_n: int = 10, min_confidence: float = 0.75, min_mask_ratio: float = 0.002):
    #     video_path = Path(video_path)

    #     # Get the first frame
    #     cap = cv2.VideoCapture(video_path)
    #     ret, frame = cap.read()
    #     if not ret:
    #         raise ValueError("Could not read video")
    #     frame = Image.fromarray(frame[:, :, ::-1])
        
    #     # Segment the people in the first frame
    #     inputs = self.processor(images=[frame], text=[prompt], return_tensors="pt").to(self.device)
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)

    #     results = self.processor.post_process_instance_segmentation(
    #         outputs,
    #         threshold=0.5,
    #         mask_threshold=0.5,
    #         target_sizes=inputs.get("original_sizes").tolist()
    #     )[0]

    #     # Filter out detections below the confidence threshold and mask size threshold
    #     total_pixels = frame.width * frame.height
    #     filtered_results = []
    #     for i in range(len(results["scores"])):
    #         score = results["scores"][i]
    #         mask = results["masks"][i]
    #         box = results["boxes"][i]
    #         mask_ratio = mask.sum() / total_pixels
    #         print(f"Score: {score}, Mask Size: {mask.sum()}, Box: {box}, Mask Ratio: {mask_ratio}")
    #         if score >= min_confidence and mask_ratio >= min_mask_ratio:
    #             filtered_results.append((score, mask, box))

    #     # Choose the top n people based on the size of the mask
    #     # Sort by mask size
    #     filtered_results.sort(key=lambda x: x[1].sum(), reverse=True)

    #     # Choose the top n people based on the size of the mask
    #     filtered_results = filtered_results[:top_n]

    #     # Render the masks as a test image
    #     test_img = np.array(frame)
    #     for i, (score, mask, box) in enumerate(filtered_results):
    #         print(f"Person {i}: Score: {score}, Mask Size: {mask.sum()}, Box: {box}, Mask Ratio: {mask.sum() / total_pixels}, mask shape: {mask.shape}")
    #         print(np.unique(mask.cpu().numpy()))
    #         # Draw an opaque mask on the test image
    #         mask_locs = np.where(mask.cpu().numpy())
    #         test_img[mask_locs] = [255, 0, 0]
            
    #     test_img = Image.fromarray(test_img)
    #     test_out_path = Path("data/test_segmentations") / video_path.stem
    #     test_out_path.mkdir(parents=True, exist_ok=True)
    #     test_img.save(test_out_path / "test_img.png")

    def segment_video(self, video_path: Path | str, prompt: str = "person", top_n: int = 10, min_confidence: float = 0.75, min_mask_ratio: float = 0.002, min_frames_visible: int = 10, max_num_frames: int | None = None, skip_frames: int = 1):
        video_path = Path(video_path)

        video_frames, _ = load_video(str(video_path))

        video_frames = video_frames[::skip_frames]

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

        # self.processor.add_inputs_to_inference_session(
        #     inference_session=inference_session,
        #     frame_idx=segment_frame,
        #     obj_ids=obj_ids,
        #     input_masks=obj_masks,
        #     original_size=(height, width),
        # )

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
        for frame_idx, frame_outputs in outputs_per_frame.items():
            frame_obj_ids = frame_outputs["object_ids"]
            frame_scores = frame_outputs["scores"]
            frame_boxes = frame_outputs["boxes"]
            frame_masks = frame_outputs["masks"]
            for i in range(len(frame_obj_ids)):
                obj_id = int(frame_obj_ids[i])
                if obj_id not in object_data:
                    object_data[obj_id] = []
                
                object_frame_data = {
                    "frame_idx": int(frame_idx),
                    "score": float(frame_scores[i]),
                    "box": [int(x) for x in frame_boxes[i]],
                    "mask": frame_masks[i],
                }
                object_data[obj_id].append(object_frame_data)

        total_frames = len(outputs_per_frame)
        obj_summary_data = {}
        total_pixels = video_frames[0].shape[0] * video_frames[0].shape[1]
        for obj_id, obj_data in object_data.items():
            # Collect proportion of frames visible, average score, average mask ratio
            visible_frames = len(obj_data)
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

        # import json
        # print(json.dumps(filtered_obj_summary_data, indent=4))
        print(filtered_obj_summary_data)

        # 1. Generate unique random colors for each tracked object
        np.random.seed(42) # For consistent colors across runs
        colors = {obj_id: np.random.randint(0, 255, (3,), dtype=np.uint8).tolist() for obj_id in filtered_obj_ids}

        # 2. Re-index filtered masks by frame index for faster rendering
        masks_by_frame = {i: [] for i in range(len(video_frames))}
        for obj_id, obj_frames in filtered_obj_data.items():
            for frame_data in obj_frames:
                frame_idx = frame_data["frame_idx"]
                mask = frame_data["mask"]
                masks_by_frame[frame_idx].append((obj_id, mask))

        # 3. Initialize OpenCV VideoWriter
        output_path = video_path.parent / f"{video_path.stem}_segmented.mp4"
        
        # Determine width and height from the first frame
        first_frame = np.array(video_frames[0])
        height, width = first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0 # Adjust if you know the exact FPS of your source
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        print(f"Writing segmented video to {output_path}...")

        # 4. Render frames
        for i, frame in enumerate(video_frames):
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

        
        
        

if __name__ == "__main__":
    splits_path = Path("data/test_splits")
    # test_split_path = splits_path / "test" / "test-Scene-031.mp4"
    test_split_path = splits_path / "soccer" / "soccer-Scene-008.mp4"
    

    segmenter = Sam3VideoPersonSegmenter()
    segmenter.segment_video(str(test_split_path), prompt="Soccer player on field", top_n=1000, max_num_frames=None, skip_frames=5)