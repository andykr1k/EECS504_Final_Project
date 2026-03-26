import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
import json
from pathlib import Path
import random
import logging
import dino_lib

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class Sam3ReIDDataset(Dataset):
    def __init__(self, root_dir: str | Path, transform=None, mask_background=True, return_dino_segmentations=False):
        """
        Args:
            root_dir: The root directory to search for '*_segmentations' folders.
            transform: torchvision transforms to apply to the cropped person images.
            mask_background: If True, sets background pixels to black using the SAM mask.
            return_dino_segmentations: If True, uses dino_lib to extract DINOv3 segmentations.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mask_background = mask_background
        self.return_dino_segmentations = return_dino_segmentations
        
        if self.return_dino_segmentations:
            self.dino_harness = dino_lib.DinoHarness()
        
        # Flat list for __getitem__: [(frame_path, mask_path, person_id, global_class_id)]
        self.samples = []
        
        # Hierarchical dictionary for the Sampler: 
        # {video_name: {slice_name: {global_class_id: [sample_idx, ...]}}}
        self.hierarchy = {}
        
        self._build_dataset()

    def _build_dataset(self):
        logging.info(f"Scanning {self.root_dir} for dataset folders...")
        segmentation_dirs = list(self.root_dir.rglob("*_segmentations"))
        
        global_class_id = 0
        total_frames = 0
        total_bboxes = 0
        
        for seg_dir in segmentation_dirs:
            # The name of the video is derived from the folder name
            video_name = seg_dir.name.replace("_segmentations", "")
            self.hierarchy[video_name] = {}
            
            # Iterate through slices
            for slice_dir in seg_dir.iterdir():
                if not slice_dir.is_dir():
                    continue
                
                slice_name = slice_dir.name
                self.hierarchy[video_name][slice_name] = {}
                
                # Iterate through all sidecars in the slice
                sidecars = list(slice_dir.glob("*_sidecar.json"))
                for sidecar_path in sidecars:
                    with open(sidecar_path, 'r') as f:
                        sidecar = json.load(f)
                    
                    frame_idx = sidecar["frame_idx"]
                    visible_ids = sidecar["visible_person_ids"]
                    
                    if not visible_ids:
                        continue
                        
                    total_frames += 1
                    
                    frame_path = slice_dir / f"{slice_name}_{frame_idx}_frame.png"
                    mask_path = slice_dir / f"{slice_name}_{frame_idx}_segmentation.npz"
                    
                    for person_id in visible_ids:
                        # Create a unique string identifier for this specific person in this slice
                        identity_key = f"{video_name}::{slice_name}::{person_id}"
                        
                        # Assign global class ID if we haven't seen this person in this slice yet
                        if identity_key not in getattr(self, '_temp_id_map', {}):
                            if not hasattr(self, '_temp_id_map'):
                                self._temp_id_map = {}
                            self._temp_id_map[identity_key] = global_class_id
                            global_class_id += 1
                            
                        current_class_id = self._temp_id_map[identity_key]
                        
                        # Add to flat list
                        sample_idx = len(self.samples)
                        self.samples.append((frame_path, mask_path, person_id, current_class_id))
                        total_bboxes += 1
                        
                        # Add to hierarchy
                        if current_class_id not in self.hierarchy[video_name][slice_name]:
                            self.hierarchy[video_name][slice_name][current_class_id] = []
                        self.hierarchy[video_name][slice_name][current_class_id].append(sample_idx)
        
        # Cleanup temp mapping
        if hasattr(self, '_temp_id_map'):
            del self._temp_id_map
            
        logging.info("--- Dataset Metadata ---")
        logging.info(f"Total Videos: {len(self.hierarchy)}")
        logging.info(f"Total Unique Slices: {sum(len(v) for v in self.hierarchy.values())}")
        logging.info(f"Total Unique Identities (Classes): {global_class_id}")
        logging.info(f"Total Frames Processed: {total_frames}")
        logging.info(f"Total Person Bounding Boxes (Samples): {total_bboxes}")
        logging.info("------------------------")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, mask_path, person_id, class_id = self.samples[idx]
        
        # 1. Load image and mask
        image_pil = Image.open(frame_path).convert("RGB")
        image = np.array(image_pil)
        mask_data = np.load(mask_path)['segmentation']
        
        # 2. Extract boolean mask for this specific person
        person_mask = (mask_data == person_id)
        
        # 3. Find bounding box to crop
        coords = np.argwhere(person_mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1  # +1 for slicing
        
        # 4. Apply mask to background (Optional but highly recommended for Re-ID)
        if self.mask_background:
            # Create a black background image, then copy only the masked pixels over
            masked_image = np.zeros_like(image)
            masked_image[person_mask] = image[person_mask]
            crop = masked_image[y0:y1, x0:x1]
        else:
            # Just take the bounding box crop
            crop = image[y0:y1, x0:x1]
            
        crop_pil = Image.fromarray(crop)
        
        if self.transform:
            crop_pil = self.transform(crop_pil)
            
        if self.return_dino_segmentations:
            with torch.no_grad():
                dino_segs = self.dino_harness.match_segmentations_to_dino([image_pil], [mask_data])
            dino_bboxes, dino_embeddings, dino_overlaps = [], [], []
            for seg in dino_segs[0]:
                if seg.person_id == person_id:
                    dino_bboxes = seg.seg_bboxes if hasattr(seg, 'seg_bboxes') else seg.dino_bboxes
                    dino_embeddings = seg.dino_embeddings
                    dino_overlaps = seg.dino_overlaps
                    break
            return crop_pil, class_id, dino_bboxes, dino_embeddings, dino_overlaps

        return crop_pil, class_id


class VideoSlicePKBatchSampler(Sampler):
    """
    Samples batches ensuring:
    1. Exact batch size of (P * K) where P is identities and K is instances per identity.
    2. One slice per video per batch to prevent ambiguous negative pairs.
    3. Maximizes hard negatives by pulling multiple identities from the same slice when available.
    """
    def __init__(self, dataset: Sam3ReIDDataset, num_identities_per_batch: int, instances_per_identity: int):
        self.hierarchy = dataset.hierarchy
        self.P = num_identities_per_batch
        self.K = instances_per_identity
        self.batch_size = self.P * self.K
        
        # Pre-calculate approximate total batches based on total samples
        self.total_samples = len(dataset)
        self.num_batches = self.total_samples // self.batch_size

    def __iter__(self):
        video_names = list(self.hierarchy.keys())
        random.shuffle(video_names)
        
        batch = []
        identities_in_batch = 0
        video_idx = 0
        
        # We loop continuously until we've exhausted our expected number of batches.
        # This acts effectively as an 'epoch'
        for _ in range(self.num_batches):
            while identities_in_batch < self.P:
                # If we run out of videos to sample from, reshuffle and reset
                if video_idx >= len(video_names):
                    random.shuffle(video_names)
                    video_idx = 0
                
                video_name = video_names[video_idx]
                video_idx += 1
                
                # Rule 1: Pick exactly ONE random slice from this video
                slice_name = random.choice(list(self.hierarchy[video_name].keys()))
                
                # Rule 2: Get all identities in this slice to create hard negatives
                slice_identities = list(self.hierarchy[video_name][slice_name].keys())
                random.shuffle(slice_identities)
                
                for identity in slice_identities:
                    if identities_in_batch >= self.P:
                        break # Batch is full of identities, wait for next batch
                    
                    instances = self.hierarchy[video_name][slice_name][identity]
                    
                    # Rule 3: Enforce K instances exactly. 
                    # If an identity has less than K frames, we sample with replacement.
                    if len(instances) >= self.K:
                        sampled_indices = random.sample(instances, self.K)
                    else:
                        sampled_indices = random.choices(instances, k=self.K)
                        
                    batch.extend(sampled_indices)
                    identities_in_batch += 1
            
            # Yield the strictly sized batch and clear for the next one
            yield batch
            batch = []
            identities_in_batch = 0

    def __len__(self):
        return self.num_batches

if __name__ == "__main__":
    # 1. Define typical Re-ID transformations (resize, random flip, normalize)
    transform = T.Compose([
        T.Resize((256, 128)), # Standard ReID aspect ratio
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Instantiate the Dataset (Preprocessing / file searching happens here)
    dataset = Sam3ReIDDataset(
        root_dir="/z/dat/person_reid/internal/input_videos",
        transform=transform,
        mask_background=True # Crops will have background blacked out based on SAM
    )

    # 3. Setup PK parameters
    P = 16 # Number of unique people per batch
    K = 4  # Number of frames per person
    # Total Batch Size = 16 * 4 = 64

    # 4. Instantiate the custom Sampler
    sampler = VideoSlicePKBatchSampler(
        dataset=dataset, 
        num_identities_per_batch=P, 
        instances_per_identity=K
    )

    # 5. Create the Dataloader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # Iterate
    for batch_idx, (images, labels) in enumerate(dataloader):
        # images shape: [64, 3, 256, 128]
        # labels shape: [64] -> e.g., [id1, id1, id1, id1, id2, id2, id2, id2, ...]
        
        # Pass to model and Circle Loss...
        print(images.shape)
        print(labels.shape)
        break