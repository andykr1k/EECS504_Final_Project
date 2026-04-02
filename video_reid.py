from dataclasses import dataclass
from pathlib import Path
import torch
import numpy as np
import imageio
import imageio.v3 as iio
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import transnetv2_pytorch as transnet
import pickle
from transformers import Sam3VideoModel, Sam3VideoProcessor
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, fcluster
import h5py

import dino_lib

SEG_ID_TYPE = np.int32
MAX_SEG_ID = np.iinfo(SEG_ID_TYPE).max

TrackletID = int  # Unique to this tracklet
SegmentationIndex = int  # Index of the segmentation in the scene
PersonID = int  # Unique person id
FrameIndex = int  # Index of the frame in the full video
SceneFrameIndex = int  # Index of the frame in the scene
SceneID = str  # Unique to this scene slice. Corresponds to the file name

SamSegmentationsNp = np.ndarray  # Shape (num_frames, H, W)
VideoFramesNp = np.ndarray  # Shape (num_frames, H, W, 3)
SegmentedDinoEmbeddings = np.ndarray  # Shape (n_overlapping_patches, dino_embedding_dim)
ContrastiveEmbedding = np.ndarray  # Shape (contrastive_embedding_dim,)

@dataclass
class SceneSlice:
    scene_id: SceneID  # Unique to this scene slice. Corresponds to the file name
    start_frame: FrameIndex  # Inclusive. When this scene starts
    end_frame: FrameIndex  # Exclusive. When this scene ends

@dataclass
class TrackletMetadata:
    tracklet_id: TrackletID  # Unique to this tracklet
    segmentation_index: SegmentationIndex  # Index of the segmentation in the scene
    person_id: PersonID | None  # Unique person id. None if unknown
    scene_id: SceneID  # Maps to the scene slice from the original video
    start_frame: SceneFrameIndex  # Inclusive. When segmentation first appears
    end_frame: SceneFrameIndex  # Exclusive. When segmentation last appears

@dataclass
class TrackletData:
    tracklet_id: TrackletID
    frame_indices: list[SceneFrameIndex]
    confidences: list[float]  # Confidence of the segmentation at each frame
    mask_sizes: list[int]  # Number of pixels in the mask at each frame
    embeddings: torch.Tensor  # Shape (num_frames, contrastive_embedding_dim)

@dataclass
class ReIDConfig:
    video_path: Path
    out_dir: Path

    sam3_prompt: str = "Child, Children, Student, Students"

    max_scene_len_frames: int = 60 * 20

    @property
    def scenes_dir(self) -> Path:
        return self.out_dir / "scenes"

    def get_scene_id(self, start_frame: FrameIndex, end_frame: FrameIndex) -> SceneID:
        return f"s{start_frame:06d}_e{end_frame:06d}"

    def get_scene_path(self, scene_id: SceneID) -> Path:
        return self.scenes_dir / f"{scene_id}.mp4"

    @property
    def segmentations_dir(self) -> Path:
        return self.out_dir / "segmentations"

    def get_segmentation_path(self, scene_id: SceneID, frame_idx: FrameIndex) -> Path:
        return self.segmentations_dir / f"{scene_id}_f{frame_idx:06d}.png"

    @property
    def tracklets_dir(self) -> Path:
        return self.out_dir / "tracklets"

    def get_tracklet_path(self, tracklet_id: TrackletID) -> Path:
        return self.tracklets_dir / f"t{tracklet_id:06d}.json"

    def get_tracklet_embeddings_path(self, tracklet_id: TrackletID) -> Path:
        return self.tracklets_dir / f"t{tracklet_id:06d}_embeddings.pkl"

@dataclass
class VideoReIDData:
    reid_config: ReIDConfig

    video_width: int
    video_height: int
    video_fps: float
    video_total_frames: int

    scene_slices: list[SceneSlice]
    tracklet_data: list[TrackletData]

# ==========================================
# Utils 1: Video Splitting
# ==========================================
# Split the original video into scene slices that we will run SAM over
# Take in a video path and a directory path that must be used exclusively for this single video
# Files are saved as `[scene_id].mp4`

def split_video(video_path: Path, out_dir: Path, max_len_frames: int) -> tuple[list[SceneSlice], float]:
    """
    Split a video both around scene cuts as found by transnet and if those go too long by
    splitting them into max_len_frames chunks.
    Saves the videos to `out_dir` with names following `get_segmentation_path`.
    If that scene name already exists we skip it.
    Returns a list of `SceneSlice` objects corresponding to the saved videos.
    """
    print("Loading TransNetV2 to GPU for scene detection...")
    model = transnet.TransNetV2()
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    
    # 1. Detect scenes
    # We use a threshold of 0.2 as defined in the reference script
    scenes = model.detect_scenes(str(video_path), threshold=0.2)
    
    scene_slices: list[SceneSlice] = []
    
    # 2. Constrain scene lengths and build SceneSlice objects
    for scene in scenes:
        # TransNetV2 returns inclusive start and end frames
        start_frame = scene['start_frame']
        # Convert to exclusive end_frame for our dataclass
        end_frame_exclusive = scene['end_frame'] + 1 
        
        current_start = start_frame
        while current_start < end_frame_exclusive:
            current_end = min(current_start + max_len_frames, end_frame_exclusive)
            
            scene_id = f"s{current_start:06d}_e{current_end:06d}"
            scene_slices.append(SceneSlice(
                scene_id=scene_id,
                start_frame=current_start,
                end_frame=current_end
            ))
            current_start = current_end

    if not scene_slices:
        print(f"No scenes found in {video_path.name}.")
        return [], 0.0

    # 3. Extract and save the video chunks
    print(f"Extracting {len(scene_slices)} scene chunks...")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame_idx = 0
    scene_idx = 0
    
    with tqdm(total=len(scene_slices), desc="Saving Scene Slices") as pbar:
        while scene_idx < len(scene_slices):
            current_scene = scene_slices[scene_idx]
            out_path = out_dir / f"{current_scene.scene_id}.mp4"
            
            # Fast-forward to the start of the current scene if we are behind.
            # Using cap.grab() is much faster than cap.read() and avoids OpenCV 
            # cap.set() keyframe seek inaccuracies.
            while current_frame_idx < current_scene.start_frame:
                ret = cap.grab()
                if not ret:
                    break
                current_frame_idx += 1
            
            # If the file already exists, skip processing it
            if out_path.exists():
                while current_frame_idx < current_scene.end_frame:
                    ret = cap.grab()
                    if not ret:
                        break
                    current_frame_idx += 1
                scene_idx += 1
                pbar.update(1)
                continue

            # File doesn't exist, we need to extract and write it
            chunk_writer = imageio.get_writer(
                str(out_path), 
                fps=fps, 
                codec='libx264', 
                pixelformat='yuv420p',
                macro_block_size=None
            )
            
            while current_frame_idx < current_scene.end_frame:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                
                # Convert BGR (OpenCV) to RGB (imageio)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                chunk_writer.append_data(frame_rgb)
                current_frame_idx += 1
                
            chunk_writer.close()
            scene_idx += 1
            pbar.update(1)

    cap.release()
    return scene_slices, fps
    

# ==========================================
# Utils 2: SAM Segmentation
# ==========================================
# For each scene slice, run SAM over all frames and save them as HDF5 with gzip compression for each slice
# Each scene segmentation is a (num_frames, H, W) numpy array with values in [0, num_segmentations_in_scene]
# For each frame run DINO and align the segmentations to the DINO embeddings and pass them through the 
# contrastive head to get the final embeddings for each segmentation on each frame

class SAMHarness:
    def __init__(self, device: str | None = None):
        """
        Initializes the SAM3 model and processor for video segmentation.
        Defaults to the best available accelerator (MPS, CUDA, or CPU).
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        
        print(f"Loading SAM3 model to {self.device}...")
        self.model = Sam3VideoModel.from_pretrained("facebook/sam3").to(self.device, dtype=torch.bfloat16)
        self.processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

def extract_segmentations(video_frames: VideoFramesNp, sam_harness: SAMHarness, prompt: str) -> tuple[SamSegmentationsNp, list[dict[SegmentationIndex, float]]]:
    """
    Runs video SAM over the given slice of frames based on a text prompt.
    
    Args:
        video_frames: Shape (num_frames, H, W, 3) 
        sam_harness: Initialized SAMHarness instance
        prompt: Text prompt to guide segmentation (e.g., "person")
        min_frame_confidence: Minimum score to accept a mask in a frame
        min_frame_mask_ratio: Minimum percentage of the image the mask must cover

    Returns:
        segmentations: Numpy array of shape (num_frames, H, W) with values 
                       in [0, num_segmentations_in_scene]. 0 is background.
        confidences: List of dictionaries mapping segmentation index to confidence score.
    """
    num_frames, height, width, _ = video_frames.shape
    total_pixels = height * width

    # The processor expects a list of frames (as numpy arrays or PIL images)
    frames_list = [video_frames[i] for i in range(num_frames)]

    # 1. Initialize the inference session with the provided frames
    inference_session = sam_harness.processor.init_video_session(
        video=frames_list,
        inference_device=sam_harness.device,
        dtype=torch.bfloat16,
    )

    # 2. Add the tracking prompt
    inference_session = sam_harness.processor.add_text_prompt(
        inference_session=inference_session,
        text=prompt,
    )

    # Setup the output structures
    # Using np.int32 to allow for high numbers of unique segmentations per scene
    segmentations = np.full((num_frames, height, width), fill_value=MAX_SEG_ID, dtype=SEG_ID_TYPE)
    confidences: list[dict[int, float]] = [{} for _ in range(num_frames)]

    # max_seg_id is reserved for the background

    # 3. Propagate and parse masks
    for model_outputs in sam_harness.model.propagate_in_video_iterator(
        inference_session=inference_session,
        show_progress_bar=False,
    ):
        frame_idx = model_outputs.frame_idx
        processed_outputs = sam_harness.processor.postprocess_outputs(inference_session, model_outputs)

        frame_obj_ids = processed_outputs["object_ids"]
        frame_scores = processed_outputs["scores"]
        frame_masks = processed_outputs["masks"]

        for j in range(len(frame_obj_ids)):
            score = float(frame_scores[j])

            # Process mask tensor
            mask_np = frame_masks[j].cpu().numpy().astype(bool)

            # Map the SAM object ID to our internal contiguous ID
            seg_idx = SEG_ID_TYPE(frame_obj_ids[j])
            if seg_idx >= MAX_SEG_ID:
                raise ValueError("Too many segmentations in this scene")

            # Write mask and score to our final structures
            # Note: In cases where masks slightly overlap, the later ID in the loop 
            # will overwrite the earlier ID for those specific pixels.
            segmentations[frame_idx][mask_np] = seg_idx
            confidences[frame_idx][seg_idx] = score

            # Write mask and score to our final structures
            # Note: In cases where masks slightly overlap, the later ID in the loop 
            # will overwrite the earlier ID for those specific pixels.
            segmentations[frame_idx][mask_np] = seg_idx
            confidences[frame_idx][seg_idx] = score

    return segmentations, confidences

def align_segmentations_to_dino(video_frames: VideoFramesNp, segmentations: SamSegmentationsNp, dino_harness: dino_lib.DinoHarness) -> list[dict[SegmentationIndex, SegmentedDinoEmbeddings]]:
    """
    Extracts the full DINO features for each frame and aligns them to the segmentations.
    Returns a list of dictionaries where each dictionary represents a frame and
    maps a segmentation index to its DINO embedding of shape (n_overlapping_patches, dino_embedding_dim)
    """
    """
    Extracts the full DINO features for each frame and aligns them to the segmentations.
    Returns a list of dictionaries where each dictionary represents a frame and
    maps a segmentation index to its DINO embedding of shape (n_overlapping_patches, dino_embedding_dim)
    """
    num_frames = video_frames.shape[0]
    batch_size = 16  # Process in batches to prevent GPU OOM
    aligned_results: list[dict[SegmentationIndex, SegmentedDinoEmbeddings]] = []
    
    for start_idx in range(0, num_frames, batch_size):
        end_idx = min(start_idx + batch_size, num_frames)
        
        frames_batch = video_frames[start_idx:end_idx]
        segs_batch = segmentations[start_idx:end_idx].copy()
        
        # 1. Prepare images: (B, H, W, C) -> list of (C, H, W) tensors
        images_list = []
        for i in range(frames_batch.shape[0]):
            # Convert to torch tensor and rearrange to Channel-First format
            img_t = torch.from_numpy(frames_batch[i]).permute(2, 0, 1)
            images_list.append(img_t)
            
        # 2. Prepare segmentations: Map 0 (SAM background) to -1 (DINO skip)
        segs_list = []
        for i in range(segs_batch.shape[0]):
            seg = segs_batch[i]
            # dino_harness ignores person_id == -1
            seg = np.where(seg == MAX_SEG_ID, -1, seg)
            segs_list.append(seg)
            
        # 3. Run DINO harness batched processing
        dino_outputs = dino_harness.match_segmentations_to_dino(images_list, segs_list)
        
        # 4. Format outputs mapping SegmentationIndex -> SegmentedDinoEmbeddings (numpy)
        for frame_dino_segs in dino_outputs:
            frame_dict = {}
            for dino_seg in frame_dino_segs:
                # Type mapping: Convert torch.Tensor to np.ndarray as defined by SegmentedDinoEmbeddings
                frame_dict[dino_seg.person_id] = dino_seg.dino_embeddings.cpu().numpy()
            
            aligned_results.append(frame_dict)
            
    return aligned_results

def extract_contrastive_embeddings(
    dino_embeddings: list[dict[SegmentationIndex, SegmentedDinoEmbeddings]], 
    contrastive_head: torch.nn.Module,
    batch_size: int = 256
) -> list[dict[SegmentationIndex, ContrastiveEmbedding]]:
    """
    Passes the DINO embeddings through the contrastive head to get the final embeddings for each segmentation on each frame.
    Returns a list of dictionaries where each dictionary represents a frame and
    maps a segmentation index to its contrastive embedding of shape (contrastive_embedding_dim,)
    """
    # Infer the device from the model
    device = next(contrastive_head.parameters()).device
    contrastive_head.eval()
    
    # 1. Flatten the nested frame/segmentation structure into a linear list
    tensor_inputs = []
    mapping = []  # To keep track of where each result goes (frame_idx, seg_idx)
    
    for frame_idx, frame_data in enumerate(dino_embeddings):
        for seg_idx, emb_np in frame_data.items():
            # Convert numpy array to torch tensor
            tensor_inputs.append(torch.from_numpy(emb_np).float())
            mapping.append((frame_idx, seg_idx))
            
    # Handle the edge case where no segmentations were found in any frame
    if not tensor_inputs:
        return [{} for _ in range(len(dino_embeddings))]
        
    # 2. Run inference in batches to prevent padding OOM
    results_np_list = []
    
    with torch.no_grad():
        for i in range(0, len(tensor_inputs), batch_size):
            # Move the current batch of tensors to the device
            batch = [t.to(device) for t in tensor_inputs[i : i + batch_size]]
            
            # Forward pass through the ReID model
            batch_out = contrastive_head(batch)
            
            # Move back to CPU and convert to numpy
            results_np_list.append(batch_out.cpu().numpy())
            
    # Concatenate all batched results into a single (total_segmentations, contrastive_dim) array
    all_results_np = np.concatenate(results_np_list, axis=0)
    
    # 3. Reconstruct the final list of dicts structure
    output: list[dict[SegmentationIndex, ContrastiveEmbedding]] = [{} for _ in range(len(dino_embeddings))]
    
    for (frame_idx, seg_idx), emb_out in zip(mapping, all_results_np):
        output[frame_idx][seg_idx] = emb_out
        
    return output

def process_scene_segmentations(
    scene_slice: SceneSlice,
    config: ReIDConfig,
    sam_harness: SAMHarness,
    dino_harness: dino_lib.DinoHarness,
    contrastive_head: torch.nn.Module
) -> tuple[SamSegmentationsNp, list[dict[SegmentationIndex, float]], list[dict[SegmentationIndex, ContrastiveEmbedding]]]:
    """
    Process the segmentations for a single scene slice.
    If segmentations are already saved, load them and skip the SAM extraction.

    Returns the segmentations, the confidences for each segmentation in each frame, 
    and the contrastive embeddings for each segmentation in each frame.
    """
    scene_id = scene_slice.scene_id
    
    # Define cache file paths
    seg_file = config.segmentations_dir / f"{scene_id}_segmentations.h5"
    conf_file = config.segmentations_dir / f"{scene_id}_confidences.pkl"
    emb_file = config.segmentations_dir / f"{scene_id}_embeddings.pkl"
    
    # 1. Check for cached results to avoid re-running heavy models
    if seg_file.exists() and conf_file.exists() and emb_file.exists():
        print(f"  -> Loading cached segmentations and embeddings for {scene_id}...")
        with h5py.File(seg_file, "r") as f:
            segmentations = f["segmentations"][:]
        
        with open(conf_file, "rb") as f:
            confidences = pickle.load(f)
            
        with open(emb_file, "rb") as f:
            contrastive_embeddings = pickle.load(f)
            
        return segmentations, confidences, contrastive_embeddings

    print(f"  -> Processing scene {scene_id} from scratch...")
    
    # 2. Load the scene's video frames into memory
    scene_path = config.get_scene_path(scene_id)
    if not scene_path.exists():
        raise FileNotFoundError(f"Video file for scene {scene_id} not found at {scene_path}")
        
    video_frames = iio.imread(scene_path, plugin="pyav")
    
    # 3. Extract SAM segmentations
    segmentations, confidences = extract_segmentations(
        video_frames=video_frames,
        sam_harness=sam_harness,
        prompt=config.sam3_prompt
    )
    
    # 4. Extract and align DINO embeddings to the masks
    dino_embeddings = align_segmentations_to_dino(
        video_frames=video_frames,
        segmentations=segmentations,
        dino_harness=dino_harness
    )
    
    # 5. Generate final contrastive embeddings for ReID clustering
    contrastive_embeddings = extract_contrastive_embeddings(
        dino_embeddings=dino_embeddings,
        contrastive_head=contrastive_head
    )
    
    # 6. Cache the results for future pipeline runs
    config.segmentations_dir.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(seg_file, "w") as f:
        f.create_dataset(
            "segmentations", 
            data=segmentations, 
            compression="gzip", 
            compression_opts=4  # Level 4 is a good balance of speed and file size reduction
        )
    
    with open(conf_file, "wb") as f:
        pickle.dump(confidences, f)
        
    with open(emb_file, "wb") as f:
        pickle.dump(contrastive_embeddings, f)
        
    return segmentations, confidences, contrastive_embeddings


# ==========================================
# Utils 3: Tracklet Generation
# ==========================================
# Track the segmentations across frames and when they drop below a certain threshold of pixels
# stop tracking and save the tracklet data. When they reappear, start a new tracklet.

def track_segmentations(
    segmentations: SamSegmentationsNp,
    confidences: list[dict[SegmentationIndex, float]],
    contrastive_embeddings: list[dict[SegmentationIndex, ContrastiveEmbedding]],
    video_width: int,
    video_height: int,
    min_frame_mask_ratio: float = 0.001,
    min_tracklet_len: int = 10,
    tracklet_start_index: int = 0,
) -> tuple[list[TrackletData], list[TrackletMetadata]]:
    """
    Track the segmentations across frames and when they drop below a certain threshold of pixels
    stop tracking and save the tracklet data. When they reappear, start a new tracklet.
    """
    num_frames = segmentations.shape[0]
    total_pixels = video_width * video_height
    
    # Active tracklets currently being built. Maps SegmentationIndex -> Tracklet dictionaries
    active_tracklets: dict[SegmentationIndex, dict] = {}
    
    final_tracklets_data: list[TrackletData] = []
    final_tracklets_meta: list[TrackletMetadata] = []
    
    current_tracklet_id = tracklet_start_index
    
    for f in range(num_frames):
        # Find unique segmentations and their pixel counts in the current frame
        seg_indices, counts = np.unique(segmentations[f], return_counts=True)
        
        valid_segs_in_frame = set()
        
        for seg_idx, count in zip(seg_indices, counts):
            if seg_idx == MAX_SEG_ID:
                continue  # Skip background
            
            ratio = count / total_pixels
            if ratio >= min_frame_mask_ratio:
                # DINO / Contrastive embedding must be present to be a valid frame
                emb = contrastive_embeddings[f].get(seg_idx)
                
                if emb is not None:
                    valid_segs_in_frame.add(seg_idx)
                    
                    if seg_idx not in active_tracklets:
                        active_tracklets[seg_idx] = {
                            "frame_indices": [],
                            "confidences": [],
                            "mask_sizes": [],
                            "embeddings": []
                        }
                    
                    # Append data for the current frame
                    active_tracklets[seg_idx]["frame_indices"].append(f)
                    active_tracklets[seg_idx]["confidences"].append(confidences[f].get(seg_idx, 0.0))
                    active_tracklets[seg_idx]["mask_sizes"].append(int(count))
                    active_tracklets[seg_idx]["embeddings"].append(emb)
        
        # Check for active tracklets that have dropped out in the current frame
        dropped_segs = []
        for active_seg in active_tracklets:
            if active_seg not in valid_segs_in_frame:
                dropped_segs.append(active_seg)
                
        # Finalize dropped tracklets
        for dropped_seg in dropped_segs:
            trk = active_tracklets.pop(dropped_seg)
            
            if len(trk["frame_indices"]) >= min_tracklet_len:
                emb_tensor = torch.tensor(np.stack(trk["embeddings"]))
                
                t_data = TrackletData(
                    tracklet_id=current_tracklet_id,
                    frame_indices=trk["frame_indices"],
                    confidences=trk["confidences"],
                    mask_sizes=trk["mask_sizes"],
                    embeddings=emb_tensor
                )
                
                t_meta = TrackletMetadata(
                    tracklet_id=current_tracklet_id,
                    segmentation_index=dropped_seg,
                    person_id=None,
                    scene_id="",  # Injected by the caller
                    start_frame=trk["frame_indices"][0],
                    end_frame=trk["frame_indices"][-1] + 1
                )
                
                final_tracklets_data.append(t_data)
                final_tracklets_meta.append(t_meta)
                current_tracklet_id += 1

    # Loop is complete. Finalize any tracklets that remained active until the very last frame
    for seg_idx, trk in active_tracklets.items():
        if len(trk["frame_indices"]) >= min_tracklet_len:
            emb_tensor = torch.tensor(np.stack(trk["embeddings"]))
            
            t_data = TrackletData(
                tracklet_id=current_tracklet_id,
                frame_indices=trk["frame_indices"],
                confidences=trk["confidences"],
                mask_sizes=trk["mask_sizes"],
                embeddings=emb_tensor
            )
            
            t_meta = TrackletMetadata(
                tracklet_id=current_tracklet_id,
                segmentation_index=seg_idx,
                person_id=None,
                scene_id="",  # Injected by the caller
                start_frame=trk["frame_indices"][0],
                end_frame=trk["frame_indices"][-1] + 1
            )
            
            final_tracklets_data.append(t_data)
            final_tracklets_meta.append(t_meta)
            current_tracklet_id += 1

    return final_tracklets_data, final_tracklets_meta


# ==========================================
# Utils 4: Build tracklet distance matrix
# ==========================================
# For each tracklet pair of tracklets, compute the angular distance between all their embeddings
# and take the pth percentile as the distance between the two tracklets.
# Alternatively, if the tracklets overlap in time, take the distance to be dist_exclusion

def compute_tracklet_distance_matrix(
    tracklet_data: list[TrackletData],
    tracklet_metadata: list[TrackletMetadata],
    dist_exclusion: float = 2.0,
    p: float = 0.9,
) -> tuple[np.ndarray, list[TrackletID]]:
    """
    Compute the distance matrix between all pairs of tracklets.
    Returns the distance matrix and a list of tracklet ids corresponding to the rows/columns.
    """
    num_tracklets = len(tracklet_data)
    dist_matrix = np.zeros((num_tracklets, num_tracklets), dtype=np.float32)
    tracklet_ids = [t.tracklet_id for t in tracklet_data]

    # 1. Pre-normalize embeddings for fast cosine similarity via dot product
    normalized_embs = []
    for t in tracklet_data:
        embs = t.embeddings.float() 
        # Normalize along the contrastive_embedding_dim
        norm_embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        normalized_embs.append(norm_embs)

    # 2. Pre-compute frame index sets for faster temporal overlap checking
    frame_sets = [set(t_data.frame_indices) for t_data in tracklet_data]
    scene_ids = [t_meta.scene_id for t_meta in tracklet_metadata]

    # 3. Compute pairwise distances
    for i in range(num_tracklets):
        for j in range(i + 1, num_tracklets):            
            if scene_ids[i] == scene_ids[j] and not frame_sets[i].isdisjoint(frame_sets[j]):
                # Temporal exclusion penalty: if they overlap in time, they cannot be the same person
                dist = dist_exclusion
            else:
                emb_i = normalized_embs[i]
                emb_j = normalized_embs[j]

                # Cosine similarity matrix between all frame embeddings of tracklet i and tracklet j
                # Shape: (N_i, N_j)
                sim_matrix = torch.mm(emb_i, emb_j.t())

                # Clamp to prevent NaN in arccos due to floating point inaccuracies (e.g., 1.0000001)
                sim_matrix = torch.clamp(sim_matrix, -1.0 + 1e-7, 1.0 - 1e-7)

                # Convert to angular distance scaled to [0, 1]
                ang_dists = torch.arccos(sim_matrix) / torch.pi

                # Flatten and take the p-th percentile
                # np.percentile expects q in [0, 100], so we multiply p by 100
                dist = float(np.percentile(ang_dists.cpu().numpy(), p * 100))

            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # The distance matrix is symmetric

    return dist_matrix, tracklet_ids


# ==========================================
# Utils 5: Cluster tracklets and apply global person ids
# ==========================================
# Use the distance matrix to cluster the tracklets.
# For each cluster, check which tracklets belong to that cluster and assign them the same person id.

def cluster_tracklets(
    distance_matrix: np.ndarray,
    tracklet_ids: list[TrackletID],
    threshold: float = 0.5,
) -> dict[TrackletID, PersonID]:
    """
    Cluster the tracklets based on the distance matrix.
    Returns a dictionary mapping each tracklet id to its person id.
    """
    # Handle edge cases where there are 0 or 1 tracklets
    if len(tracklet_ids) == 0:
        return {}
    if len(tracklet_ids) == 1:
        return {tracklet_ids[0]: 0}

    # Convert the 2D square distance matrix into a 1D condensed distance matrix for scipy
    condensed_dist = ssd.squareform(distance_matrix)

    # 2. Perform Hierarchical Clustering
    # 'complete' linkage ensures that the maximum distance between ANY two tracklets in a cluster 
    # is less than or equal to the threshold. This perfectly respects the temporal exclusion penalty.
    Z = linkage(condensed_dist, method='complete')

    # 3. Flatten the hierarchical tree into distinct clusters based on the distance threshold
    # criterion='distance' means it forms clusters where no two elements exceed the distance threshold
    cluster_labels = fcluster(Z, t=threshold, criterion='distance')

    # 4. Map the tracklet IDs back to a 0-indexed Person ID
    # fcluster returns 1-indexed labels (1, 2, 3...), so we subtract 1
    tracklet_to_person_id: dict[TrackletID, PersonID] = {}
    for idx, tracklet_id in enumerate(tracklet_ids):
        tracklet_to_person_id[tracklet_id] = int(cluster_labels[idx] - 1)

    return tracklet_to_person_id


# ==========================================
# Putting it all together
# ==========================================
def generate_person_colors(person_ids: set[PersonID]) -> dict[PersonID, np.ndarray]:
    """Generates a unique RGB color for each Person ID using a matplotlib colormap."""
    cmap = plt.get_cmap("tab20")
    # Exclude None (which might represent background or unassigned)
    valid_ids = [pid for pid in person_ids if pid is not None]
    return {
        pid: (np.array(cmap(i % 20)[:3]) * 255).astype(np.uint8)
        for i, pid in enumerate(valid_ids)
    }

def run_video_reid_pipeline(
    config: ReIDConfig,
    sam_harness: SAMHarness,
    dino_harness: dino_lib.DinoHarness,
    contrastive_head: torch.nn.Module
) -> VideoReIDData:
    """
    Executes the full Video ReID pipeline end-to-end.
    """
    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.scenes_dir.mkdir(exist_ok=True)
    config.segmentations_dir.mkdir(exist_ok=True)
    config.tracklets_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------
    # 1. Split into scenes
    # ---------------------------------------------------------
    print("Splitting video into scenes...")
    scene_slices, fps = split_video(config.video_path, config.scenes_dir, config.max_scene_len_frames)
    
    # Grab video dimensions from the first scene to populate config
    first_scene_path = config.get_scene_path(scene_slices[0].scene_id)
    sample_frame = iio.imread(first_scene_path, index=0, plugin="pyav")
    video_height, video_width, _ = sample_frame.shape

    all_tracklets_data: list[TrackletData] = []
    all_tracklets_meta: list[TrackletMetadata] = []
    global_tracklet_idx = 0

    # ---------------------------------------------------------
    # 2 & 3. Extract Segmentations, Embeddings, and Tracklets
    # ---------------------------------------------------------
    print("Processing scenes (SAM + DINO + Tracking)...")
    for scene in tqdm(scene_slices, desc="Scenes"):
        # Process segmentations (Returns cached versions if already run)
        segmentations, confidences, embeddings = process_scene_segmentations(
            scene_slice=scene,
            config=config,
            sam_harness=sam_harness,
            dino_harness=dino_harness,
            contrastive_head=contrastive_head
        )

        # Track segmentations to form unbroken tracklets
        tracklets_data, tracklets_meta = track_segmentations(
            segmentations=segmentations,
            confidences=confidences,
            contrastive_embeddings=embeddings,
            video_width=video_width,
            video_height=video_height,
            tracklet_start_index=global_tracklet_idx
        )

        # Inject scene context into metadata
        for meta in tracklets_meta:
            meta.scene_id = scene.scene_id
            
        all_tracklets_data.extend(tracklets_data)
        all_tracklets_meta.extend(tracklets_meta)
        global_tracklet_idx += len(tracklets_data)

    # ---------------------------------------------------------
    # 4. Build Distance Matrix
    # ---------------------------------------------------------
    print("Computing tracklet distance matrix...")
    distance_matrix, tracklet_ids_order = compute_tracklet_distance_matrix(
        tracklet_data=all_tracklets_data,
        tracklet_metadata=all_tracklets_meta,
        dist_exclusion=2.0,  # Ensure epsilon_neg > max possible angular distance
        p=0.9
    )

    # ---------------------------------------------------------
    # 5. Cluster Tracklets
    # ---------------------------------------------------------
    print("Clustering tracklets to assign Global Person IDs...")
    similarity_threshold = 0.25  # Cosine similarity margin within which detections are considered the same person
    distance_threshold = np.arccos(similarity_threshold) / np.pi
    tracklet_to_person_id = cluster_tracklets(
        distance_matrix=distance_matrix,
        tracklet_ids=tracklet_ids_order,
        threshold=distance_threshold
    )

    # ---------------------------------------------------------
    # 6. Backward Assignment of Person IDs
    # ---------------------------------------------------------
    print("Assigning Person IDs back to metadata...")
    for meta in all_tracklets_meta:
        meta.person_id = tracklet_to_person_id.get(meta.tracklet_id, None)

    # Construct the final data object
    reid_data = VideoReIDData(
        reid_config=config,
        video_width=video_width,
        video_height=video_height,
        video_fps=fps,
        video_total_frames=sum(s.end_frame - s.start_frame for s in scene_slices),
        scene_slices=scene_slices,
        tracklet_data=all_tracklets_data
    )

    # ---------------------------------------------------------
    # 7. Render Final Video
    # ---------------------------------------------------------
    print("Rendering final annotated video...")
    output_video_path = config.out_dir / "final_reid_output.mp4"
    
    # Generate unique colors for each person
    unique_person_ids = set(m.person_id for m in all_tracklets_meta)
    person_colors = generate_person_colors(unique_person_ids)

    # Group metadata by scene for fast O(1) lookups during rendering
    scene_metadata_map = {scene.scene_id: [] for scene in scene_slices}
    for meta in all_tracklets_meta:
        scene_metadata_map[meta.scene_id].append(meta)

    # Use imageio writer to stream frames to disk and avoid OOM
    with imageio.get_writer(output_video_path, fps=fps, macro_block_size=None) as writer:
        for scene in tqdm(scene_slices, desc="Rendering Scenes"):
            scene_path = config.get_scene_path(scene.scene_id)
            
            segmentations, _, _ = process_scene_segmentations(
                scene_slice=scene,
                config=config,
                sam_harness=sam_harness,
                dino_harness=dino_harness,
                contrastive_head=contrastive_head
            )
            
            # Build a mapping of SegmentationIndex -> {color, person_id} for this scene
            scene_metas = scene_metadata_map[scene.scene_id]
            seg_idx_to_info = {}
            for meta in scene_metas:
                if meta.person_id is not None:
                    seg_idx_to_info[meta.segmentation_index] = {
                        "color": person_colors[meta.person_id],
                        "person_id": meta.person_id
                    }

            # Read video frames for this scene
            scene_frames = iio.imread(scene_path, plugin="pyav")
            
            for frame_idx, frame in enumerate(scene_frames):
                # Using np.copy() to ensure it is writeable for OpenCV drawing
                annotated_frame = np.copy(frame)
                frame_segs = segmentations[frame_idx] # Shape: (H, W)

                # Overlay masks and text
                for seg_idx, info in seg_idx_to_info.items():
                    color = info["color"]
                    person_id = info["person_id"]

                    # Create boolean mask for this specific segmentation index
                    mask = (frame_segs == seg_idx)
                    
                    # If this segmentation isn't in the current frame, skip
                    if not np.any(mask):
                        continue
                    
                    # Apply color overlay (alpha blending: 50% original, 50% mask color)
                    annotated_frame[mask] = (annotated_frame[mask] * 0.5 + color * 0.5).astype(np.uint8)
                    
                    # --- Text Overlay Logic ---
                    # Find coordinates to place the text (using top-center of the mask)
                    y_coords, x_coords = np.where(mask)
                    top_y = np.min(y_coords)
                    center_x = int(np.mean(x_coords))
                    
                    text = f"ID: {person_id}"
                    
                    # Calculate text size to properly center it
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    
                    # Place text slightly above the mask, keeping it within frame bounds
                    text_x = max(0, min(center_x - (text_width // 2), video_width - text_width))
                    text_y = max(text_height, top_y - 10)
                    text_pos = (text_x, text_y)

                    # Draw a black outline for readability, then the white text
                    cv2.putText(annotated_frame, text, text_pos, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                    cv2.putText(annotated_frame, text, text_pos, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                
                writer.append_data(annotated_frame)

if __name__ == "__main__":
    config = ReIDConfig(
        video_path=Path("/z/dat/person_reid/val/internal/input_videos/classroom/musical_chair_game.mp4"),
        out_dir=Path("/scratch4/home/adempst/projects/EECS504_Final_Project/data/musical_chair_game_output"),
    )
    sam_harness = SAMHarness("cuda")
    dino_harness = dino_lib.DinoHarness(device="cuda", checkpoint="facebook/dinov3-vitl16-pretrain-lvd1689m")

    head_weights_path = Path("/scratch4/home/adempst/projects/EECS504_Final_Project/checkpoints/New_metrics_large_dino/best_model.pth")
    from train import ReIDTransformerModel
    contrastive_head = ReIDTransformerModel(dino_dim=1024)
    contrastive_head.load_state_dict(torch.load(head_weights_path))
    contrastive_head = contrastive_head.to("cuda")
    contrastive_head.eval()

    run_video_reid_pipeline(config, sam_harness, dino_harness, contrastive_head)