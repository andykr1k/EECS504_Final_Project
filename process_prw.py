import os
import json
import logging
import numpy as np
import scipy.io as sio
from pathlib import Path
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

import torch
from transformers import Sam3Processor, Sam3Model

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class Sam3PersonSegmenter:
    def __init__(self, device: str | None = None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    def get_masks(self, image_path: Path, prompt: str) -> list[np.ndarray]:
        image = Image.open(image_path)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]  # We only passed in a single image so we can take the first element

        masks = results["masks"]
        masks = [mask.cpu().numpy() for mask in masks]
        return masks

def get_annotation_matrix(mat_path: Path) -> np.ndarray:
    """
    Safely extracts the bounding box matrix from a PRW .mat file.
    PRW .mat files typically contain a key like 'box_new' or 'anno'.
    """
    mat_data = sio.loadmat(str(mat_path))
    for key in mat_data:
        # Ignore MATLAB metadata keys
        if not key.startswith('__'): 
            data = mat_data[key]
            if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 5:
                return data
    return np.array([])

def preprocess_prw_for_sam3_reid(prw_root_dir: Path, output_dir: Path):
    """
    Preprocesses the PRW dataset into a frame-sliced identity format.
    """
    sam3_segmenter = Sam3PersonSegmenter()

    prw_root_dir = Path(prw_root_dir).resolve()
    output_dir = Path(output_dir).resolve()
    
    frames_dir = prw_root_dir / "frames"
    annotations_dir = prw_root_dir / "annotations"
    
    if not frames_dir.exists() or not annotations_dir.exists():
        raise FileNotFoundError(f"Ensure {frames_dir} and {annotations_dir} exist.")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track the sequential frame index for each identity
    # e.g., identity_seq_counts[45] = 3 (means we are on the 3rd frame for ID 45)
    identity_seq_counts = defaultdict(int)
    
    # Get all frames
    frame_paths = sorted(list(frames_dir.glob("*.jpg")))
    total_frames = len(frame_paths)
    logging.info(f"Found {total_frames} frames. Beginning processing...")

    for i, frame_path in enumerate(tqdm(frame_paths)):
        frame_name = frame_path.name
        
        # PRW annotations match the frame name but add .mat
        mat_path = annotations_dir / f"{frame_name}.mat"
        if not mat_path.exists():
            logging.warning(f"Missing annotation for {frame_name}, skipping.")
            continue

        # Extract annotations. PRW format is typically [id, x, y, w, h] 
        # (Adjust indices below if your specific PRW distribution uses [x, y, w, h, id])
        anno_matrix = get_annotation_matrix(mat_path)
        if anno_matrix.size == 0:
            continue
            
        valid_targets = []
        for row in anno_matrix:
            person_id = int(row[0])
            if person_id == -2: # Skip unknown/ambiguous identities
                continue
                
            x, y, w, h = map(int, row[1:5])
            valid_targets.append((person_id, x, y, w, h))

        if not valid_targets:
            continue # No valid known identities in this frame

        # Run SAM3 only once per frame
        sam_masks = sam3_segmenter.get_masks(frame_path, "person") # "person, man, woman, child, pedestrian, people")
        if not sam_masks:
            continue
            
        # We need the image dimensions to create boolean bounding box masks
        # Assuming all sam_masks share the same shape (H, W)
        img_h, img_w = sam_masks[0].shape 

        # Process each valid ground truth bounding box
        for person_id, x, y, w, h in valid_targets:
            # Constrain bbox to image dimensions
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)
            
            # Create a boolean mask for the bounding box
            bbox_mask = np.zeros((img_h, img_w), dtype=bool)
            bbox_mask[y1:y2, x1:x2] = True
            bbox_area = bbox_mask.sum()
            
            if bbox_area == 0:
                continue

            best_mask = None
            highest_iobb = 0.0
            
            # Find the SAM mask that best fits inside this bounding box
            for mask in sam_masks:
                mask_area = mask.sum()
                if mask_area == 0:
                    continue
                    
                intersection = np.logical_and(mask, bbox_mask).sum()
                iom = intersection / mask_area       # Intersection over Mask
                iobb = intersection / bbox_area      # Intersection over BBox
                
                # Strict spatial requirements:
                # 1. 90%+ of the mask must be inside the bounding box
                # 2. The mask must cover at least 15% of the bounding box (prevents tiny artifact matching)
                if iom > 0.90 and iobb > 0.15:
                    if iobb > highest_iobb:
                        highest_iobb = iobb
                        best_mask = mask

            if best_mask is not None:
                # We have a valid match! Prepare the output directory.
                seq_idx = identity_seq_counts[person_id]
                identity_seq_counts[person_id] += 1
                
                # prw_identity_1_segmentations/dummy_slice/
                identity_dir = output_dir / f"prw_identity_{person_id}_segmentations" / "dummy_slice"
                identity_dir.mkdir(parents=True, exist_ok=True)
                
                base_filename = f"dummy_slice_{seq_idx}"
                
                # 1. Symlink the frame (Saves massive amounts of disk space)
                symlink_path = identity_dir / f"{base_filename}_frame.png"
                if not symlink_path.exists():
                    os.symlink(frame_path, symlink_path)
                
                # 2. Generate and save the segmentation mask
                # Background = 255, Person = 0
                out_mask = np.full((img_h, img_w), 255, dtype=np.uint8)
                mask_locations = np.where(best_mask)
                out_mask[mask_locations] = 0 
                
                mask_path = identity_dir / f"{base_filename}_segmentation.png"
                Image.fromarray(out_mask).save(mask_path)
                
                # 3. Generate the sidecar JSON
                sidecar_path = identity_dir / f"{base_filename}_sidecar.json"
                sidecar_data = {
                    "frame_idx": seq_idx,
                    "visible_person_ids": [0]
                }
                with open(sidecar_path, 'w') as f:
                    json.dump(sidecar_data, f, indent=4)

        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1}/{total_frames} frames...")

    logging.info("Processing complete.")
    logging.info(f"Total Unique Identities Extracted: {len(identity_seq_counts)}")

# Example Execution
if __name__ == "__main__":
    PRW_ROOT = Path("/z/dat/person_reid_in_the_wild")
    OUTPUT_DIR = Path("/z/dat/person_reid/train/person_reid_in_the_wild")
    
    preprocess_prw_for_sam3_reid(PRW_ROOT, OUTPUT_DIR)