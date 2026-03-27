import torch
from torch.utils.data import Dataset, Sampler, DataLoader, default_collate
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from PIL import Image
import numpy as np
import json
from pathlib import Path
import random
import logging
import dino_lib
import torchvision.transforms.functional as F

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class Sam3ReIDDataset(Dataset):
    def __init__(self, root_dir: str | Path, transform=None, mask_background=True):
        """
        Args:
            root_dir: The root directory to search for '*_segmentations' folders.
            transform: torchvision transforms to apply to the cropped person images.
            mask_background: If True, sets background pixels to black using the SAM mask.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mask_background = mask_background
        
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

    def __getitem__(self, index):
        frame_path, mask_path, person_id, class_id = self.samples[index]
        
        # 1. Extract image logic locally to exploit num_workers
        image_pil = Image.open(frame_path).convert("RGB")
        image_np = np.array(image_pil)
        
        # 2. Fix np.load Memory Leak using a context manager
        with np.load(mask_path) as data:
            mask_data = data['segmentation']
            
        person_mask = (mask_data == person_id)
        
        # 3. Optimize NumPy Masking using fast vectorized broadcasting
        if self.mask_background:
            out_img = image_np * person_mask[..., None]
        else:
            out_img = image_np
            
        out_pil = Image.fromarray(out_img)
        
        # 4. Perfectly align spatial transformations for both image and mask
        if self.transform:
            mask_tensor = tv_tensors.Mask(torch.from_numpy(person_mask))
            out_img_transformed, out_mask_transformed = self.transform(out_pil, mask_tensor)
        else:
            out_img_transformed = F.to_tensor(out_pil)
            out_mask_transformed = person_mask
            
        return frame_path, mask_path, person_id, class_id, image_pil, mask_data, out_img_transformed, out_mask_transformed

def reid_collate_fn(batch):
    frame_paths = [item[0] for item in batch]
    mask_paths = [item[1] for item in batch]
    person_ids = [item[2] for item in batch]
    labels = default_collate([item[3] for item in batch])
    
    original_images = [item[4] for item in batch]
    original_masks = [item[5] for item in batch]

    transformed_images = [item[6] for item in batch]
    transformed_masks = [item[7] for item in batch]
    
    return frame_paths, mask_paths, person_ids, labels, original_images, original_masks, transformed_images, transformed_masks

class DinoDataLoaderWrapper:
    """
    Wraps a torch DataLoader to compute DINO embeddings in batches.
    Requires use of reid_collate_fn in the underlying DataLoader.
    """
    def __init__(self, dataloader, dino_harness=None, device="cuda"):
        self.dataloader = dataloader
        self.dino_harness = dino_harness
        self.device = device
        self._dino_initialized = False

    def _init_dino(self):
        if not self._dino_initialized:
            if self.dino_harness is None:
                import dino_lib
                self.dino_harness = dino_lib.DinoHarness(device=self.device)
            self._dino_initialized = True

    def __iter__(self):
        self._init_dino()
        for batch in self.dataloader:
            frame_paths, mask_paths, person_ids, labels, original_images, original_masks, images, masks = batch
            
            # Extract unique frames from the batch for DINO
            unique_frames = {}
            for f_path, orig_img, orig_mask in zip(frame_paths, original_images, original_masks):
                if str(f_path) not in unique_frames:
                    unique_frames[str(f_path)] = (orig_img, orig_mask)
            
            unique_f_paths = list(unique_frames.keys())
            u_images = [unique_frames[p][0] for p in unique_f_paths]
            u_masks = [unique_frames[p][1] for p in unique_f_paths]
            
            with torch.no_grad():
                assert self.dino_harness is not None
                print("Dino running")
                u_dino_segs = self.dino_harness.match_segmentations_to_dino(u_images, u_masks)
                print("Dino finished")
            
            dino_segs_by_path = {p: segs for p, segs in zip(unique_f_paths, u_dino_segs)}
            
            batch_bboxes = []
            batch_embeddings = []
            batch_overlaps = []
            
            batch_size = len(labels)
            for i in range(batch_size):
                f_path = str(frame_paths[i])
                p_id = person_ids[i]
                dino_segs = dino_segs_by_path[f_path]
                
                b_bboxes, b_embeddings, b_overlaps = [], [], []
                for seg in dino_segs:
                    if seg.person_id == p_id:
                        b_bboxes = seg.seg_bboxes if hasattr(seg, 'seg_bboxes') else seg.dino_bboxes
                        b_embeddings = seg.dino_embeddings
                        b_overlaps = seg.dino_overlaps
                        break
                batch_bboxes.append(b_bboxes)
                batch_embeddings.append(b_embeddings)
                batch_overlaps.append(b_overlaps)
            
            yield images, labels, masks, batch_bboxes, batch_embeddings, batch_overlaps

    def __len__(self):
        return len(self.dataloader)


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
    # 1. Define typical Re-ID transformations
    # Using torchvision.transforms.v2 to sync spatial transforms across image and mask
    transform = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    # Note: reid_collate_fn handles fallback if PIL images/arrays cannot be default stacked.
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=reid_collate_fn
    )

    dino_dataloader = DinoDataLoaderWrapper(dataloader)

    # Iterate
    for batch_idx, (images, labels, masks, batch_bboxes, batch_embeddings, batch_overlaps) in enumerate(dino_dataloader):
        # images shape: [64, 3, height, width] based on transform/original
        # labels shape: [64] -> e.g., [id1, id1, id1, id1, id2, id2, id2, id2, ...]
        
        # Pass to model and Circle Loss...
        print(len(images), images[0].shape)
        print(labels.shape)
        print(len(masks), masks[0].shape)
        break

    # Sample batches as fast as possible to test the speed
    import time
    from tqdm import tqdm
    sample_test_len_s = 120
    start_time = time.time()
    for batch_idx, (images, labels, masks, batch_bboxes, batch_embeddings, batch_overlaps) in enumerate(tqdm(dino_dataloader)):
        if time.time() - start_time > sample_test_len_s:
            break
    print(f"Sampled {batch_idx + 1} batches in {time.time() - start_time} seconds. {batch_idx / (time.time() - start_time)} batches/second")