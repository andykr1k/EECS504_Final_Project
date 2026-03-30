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
from torchvision.io import read_image
from codetiming import Timer

class Sam3ReIDDataset(Dataset):
    def __init__(self, root_dir: str | Path, transform=None, target_size=(1080, 1920)):
        """
        Args:
            root_dir: The root directory to search for '*_segmentations' folders.
            transform: torchvision transforms to apply to the cropped person images.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size
        self.resize_transform = v2.Resize(self.target_size, antialias=True)

        # Flat list for __getitem__: [(frame_path, mask_path, person_id, global_class_id)]
        self.samples = []

        # Hierarchical dictionary for the Sampler: 
        # {video_name: {slice_name: {global_class_id: [sample_idx, ...]}}}
        self.hierarchy = {}

        self._build_dataset()

    @Timer(name="Build Dataset", text="Build Dataset: {:.4f} seconds", logger=None)
    def _build_dataset(self):
        logging.info(f"Scanning {self.root_dir} for dataset folders...")
        global_id_map = {}

        segmentation_dirs = list(self.root_dir.rglob("*_segmentations"))

        global_class_id = 0
        total_frames = 0
        total_bboxes = 0

        for seg_dir in segmentation_dirs:
            video_name = seg_dir.name.replace("_segmentations", "")
            assert video_name not in self.hierarchy, f"Duplicate video name: {video_name}"
            self.hierarchy[video_name] = {}

            for slice_dir in seg_dir.iterdir():
                if not slice_dir.is_dir():
                    print(f"WARNING: {slice_dir} is not a directory, skipping")
                    continue
                
                slice_name = slice_dir.name
                assert slice_name not in self.hierarchy[video_name], f"Duplicate slice name: {slice_name}"
                self.hierarchy[video_name][slice_name] = {}

                sidecar_paths = list(slice_dir.glob("*_sidecar.json"))
                # Sort sidecar paths by frame index
                sidecar_paths.sort(key=lambda x: int(x.name.split("_")[-2]))
                for sidecar_path in sidecar_paths:
                    with open(sidecar_path, 'r') as f:
                        sidecar = json.load(f)
                
                    frame_idx = sidecar["frame_idx"]
                    visible_ids = sidecar["visible_person_ids"]
                    
                    if not visible_ids:
                        continue

                    total_frames += 1

                    frame_path = slice_dir / f"{slice_name}_{frame_idx}_frame.png"
                    # mask_path = slice_dir / f"{slice_name}_{frame_idx}_segmentation.npz"
                    # Changing to masks as pngs
                    mask_path = slice_dir / f"{slice_name}_{frame_idx}_segmentation.png"

                    for person_idx in visible_ids:
                        # We use a unique identifier to define the positive set
                        identity_key = f"{video_name}::{slice_name}::{person_idx}"

                        # However, for torch we need a numerical ID
                        if identity_key not in global_id_map:
                            global_id_map[identity_key] = global_class_id
                            global_class_id += 1
                        current_class_id = global_id_map[identity_key]

                        # Now we can add the sample to the flat list
                        sample_idx = len(self.samples)
                        self.samples.append((frame_path, mask_path, person_idx, current_class_id))
                        total_bboxes += 1

                        # Add to hierarchy
                        if current_class_id not in self.hierarchy[video_name][slice_name]:
                            self.hierarchy[video_name][slice_name][current_class_id] = []
                        self.hierarchy[video_name][slice_name][current_class_id].append(sample_idx)

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
        frame_path, mask_path, person_idx, class_id = self.samples[index]

        # Use torchvision to efficiently load the image
        with Timer("Read Image", text="Read Image: {:.4f} seconds", logger=None):
            image = read_image(str(frame_path))
            image_tensor = tv_tensors.Image(image)
        
        # Load the mask
        with Timer("Load Mask", text="Load Mask: {:.4f} seconds", logger=None):
            # with np.load(mask_path) as data:
            #     mask_data = data['segmentation']
            # person_mask = (mask_data == person_idx)
            # mask_tensor = tv_tensors.Mask(torchch.from_numpy(person_mask))

            # read_image reads 8-bit grayscale PNGs as a (1, H, W) uint8 tensor
            raw_mask_data = read_image(str(mask_path))
            mask_data = raw_mask_data.squeeze(0)
            person_mask = (mask_data == person_idx)
            mask_tensor = tv_tensors.Mask(person_mask)

        with Timer("Resize Image", text="Resize Image: {:.4f} seconds", logger=None):
            image_tensor, mask_tensor = self.resize_transform(image_tensor, mask_tensor)

        # Apply the transforms to the image and mask
        if self.transform:
            with Timer("Apply Transforms", text="Apply Transforms: {:.4f} seconds", logger=None):
                image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        image_tensor = F.convert_image_dtype(image_tensor, dtype=torch.float32)

        return {
            "person_id": person_idx,
            "class_id": class_id,
            "image": image_tensor.contiguous(),
            "mask": mask_tensor.contiguous()
        }

def reid_collate_fn(batch):
    person_ids = default_collate([item["person_id"] for item in batch])
    class_ids = default_collate([item["class_id"] for item in batch])
    images = default_collate([item["image"] for item in batch])
    masks = default_collate([item["mask"] for item in batch])

    return {
        "person_id": person_ids,
        "class_id": class_ids,
        "image": images,
        "mask": masks
    }

class DinoDataLoaderWrapper:
    """
    Wraps a torch DataLoader around a Sam3ReIDDataset to compute the DINO embeddings for each batch in parallel.
    """
    dino_harness: dino_lib.DinoHarness | None = None

    def __init__(
        self,
        dataloader,
        transform=None,
        dino_harness=None,
        device="cuda",
        checkpoint="facebook/dinov3-vits16-pretrain-lvd1689m",
        max_side_len=1024,
    ):
        self.dataloader = dataloader
        self.transform = transform
        self.dino_harness = dino_harness
        self.device = device
        self.checkpoint = checkpoint
        self.max_side_len = max_side_len
        self._init_dino()
    
    @Timer(name="Init DINO", text="Init DINO: {:.4f} seconds", logger=None)
    def _init_dino(self):
        if self.dino_harness is None:
            self.dino_harness = dino_lib.DinoHarness(device=self.device, checkpoint=self.checkpoint, max_side_len=self.max_side_len)
        self.dino_harness.model.eval()

    def __iter__(self):
        for batch in self.dataloader:
            # Convert the images to PIL to prepare to pass to DINO
            with Timer("Convert images for DINO", text=f"Convert {len(batch['image'])} images for DINO:" + " {:.4f} seconds", logger=None):
                images = batch["image"]
                masks = batch["mask"]

            # if self.transform is not None:
            #     with Timer("Apply Transforms", text="Apply Transforms: {:.4f} seconds", logger=None):
            #         images = tv_tensors.Image(images).to(device=self.device)
            #         masks = tv_tensors.Mask(masks).to(device=self.device)
            #         images, masks = self.transform(images, masks)

            if self.transform is not None:
                with Timer("Apply Transforms", text="Apply Transforms: {:.4f} seconds", logger=None):
                    # 1. Move the raw batch to the GPU
                    images = tv_tensors.Image(images).to(device=self.device)
                    masks = tv_tensors.Mask(masks).to(device=self.device)
                    
                    # 2. Iterate through the batch on the GPU to force independent random rolls
                    transformed_images = []
                    transformed_masks = []
                    
                    for img, mask in zip(images, masks):
                        img = tv_tensors.Image(img)
                        mask = tv_tensors.Mask(mask)
                        t_img, t_mask = self.transform(img, mask)
                        transformed_images.append(t_img)
                        transformed_masks.append(t_mask)
                    
                    # 3. Stack them back into batched tensors
                    images = torch.stack(transformed_images)
                    masks = torch.stack(transformed_masks)

            with torch.no_grad():
                assert self.dino_harness is not None
                with Timer("Match Segmentations to DINO", text="Match Segmentations to DINO: {:.4f} seconds", logger=None):
                    dino_segmentations = self.dino_harness.match_bool_segmentations_to_dino(images, masks)

            with Timer("Process DINO Segmentations", text="Process DINO Segmentations: {:.4f} seconds", logger=None):
                person_ids, class_ids, bboxes, embeddings, overlaps = [], [], [], [], []
                for batch_index, dino_segmentation in enumerate(dino_segmentations):
                    # The segmentations can include multiple objects, but we only use one so we expect there to only be one segmentation
                    if len(dino_segmentation) == 0:
                        print(f"Expected 1 segmentation per image, got {len(dino_segmentation)} for identities {[seg.person_id for seg in dino_segmentation]} for batch index {batch_index}")
                        continue
                    dino_segmentation = dino_segmentation[0]

                    person_id = batch["person_id"][batch_index]
                    class_id = batch["class_id"][batch_index]

                    person_ids.append(person_id)
                    class_ids.append(class_id)
                    bboxes.append(dino_segmentation.dino_bboxes)
                    embeddings.append(dino_segmentation.dino_embeddings)
                    overlaps.append(dino_segmentation.dino_overlaps)


            assert len(person_ids) == len(class_ids) == len(bboxes) == len(embeddings) == len(overlaps) == len(images) == len(masks), f"Length mismatch: {len(person_ids)} {len(class_ids)} {len(bboxes)} {len(embeddings)} {len(overlaps)} {len(images)} {len(masks)}"
            yield {
                "person_id": person_ids,
                "class_id": class_ids,
                "image": images,
                "mask": masks,
                # For these we cannot convert to tensors as each image has a different number of overlapping bboxes
                "bboxes": bboxes,
                "embeddings": embeddings,
                "overlaps": overlaps
            }

    def __len__(self):
        return len(self.dataloader)

class VideoSlicePKBatchSampler(Sampler):
    """
    Samples batches ensuring:
    1. Extract batch size of (P * K) where P is the number of identities and K is the instances per identity.
    2. Only one slice per video per batch to prevent negatives pairs that are actually from the same person from different scenes.
    3. When possible, pull desired_identies_per_video identities from the same slice to maximize hard negatives.
    """
    def __init__(self, dataset: Sam3ReIDDataset, num_identities: int, instances_per_identity: int, desired_identies_per_video: int = 2):
        self.heirarchy = dataset.hierarchy
        self.P = num_identities
        self.K = instances_per_identity
        self.desired_identies_per_video = desired_identies_per_video

        self.batch_size = self.P * self.K

        self.total_samples = len(dataset)
        self.num_batches = self.total_samples // self.batch_size
        print(f"Initialized VideoSlicePKBatchSampler with P={self.P}, K={self.K}, batch_size={self.batch_size}, num_batches={self.num_batches}")

        self.video_order_cache = []
    
    def _sample_video(self):
        """
        Returns a random video, but ensures that we sample all videos equally over time.
        """
        if len(self.video_order_cache) == 0:
            self.video_order_cache = list(self.heirarchy.keys())
            random.shuffle(self.video_order_cache)
        
        return self.video_order_cache.pop()

    def __iter__(self):
        for _ in range(self.num_batches):
            with Timer("Sample PK", text="Sample PK: {:.4f} seconds", logger=None):
                batch = []
                identities_in_batch = 0

                # Cache the slice choices in case we need to pull more identities from the same slice
                slice_choice_cache = {}  # video_name -> slice_name

                while identities_in_batch < self.P:
                    video_name = self._sample_video()
                    video_data = self.heirarchy[video_name]

                    # Pick one random slice from the video
                    if video_name in slice_choice_cache:
                        slice_name = slice_choice_cache[video_name]
                    else:
                        slice_name = random.choice(list(video_data.keys()))
                        slice_choice_cache[video_name] = slice_name
                    slice_data = video_data[slice_name]

                    # Get the identities in the slice
                    slice_identities = list(slice_data.keys())
                    valid_slice_identities = [identity for identity in slice_identities if len(slice_data[identity]) >= self.K]
                    random.shuffle(valid_slice_identities)

                    # Choose a set of identities to pull from this slice
                    num_identities_to_pull = min(len(valid_slice_identities), self.P - identities_in_batch, self.desired_identies_per_video)
                    identities_to_pull = valid_slice_identities[:num_identities_to_pull]

                    for identity in identities_to_pull:
                        instances = slice_data[identity]

                        # We sample with enforced diversity by spacing out the samples over the entire video
                        N = len(instances)
                        spaced_samples = [
                            random.choice(instances[i * N // self.K : (i + 1) * N // self.K]) 
                            for i in range(self.K)
                        ]

                        batch.extend(spaced_samples)
                        identities_in_batch += 1

            yield batch
    
    def __len__(self):
        return self.num_batches

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    # 1. Define typical Re-ID transformations
    # Using torchvision.transforms.v2 to sync spatial transforms across image and mask
    transform = v2.Compose([
    # --- 1. Safe Spatial Transforms ---
        v2.RandomHorizontalFlip(p=0.5),
        
        # Slight rotation (5 degrees), translation (5%), and scaling (95% to 105%)
        # The mask will perfectly track with these changes.
        v2.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),

        # --- 2. Color and Lighting (Photometric) ---
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        v2.RandomGrayscale(p=0.1),
        
        # Randomly apply Gaussian Blur to 10% of images to simulate poor focus
        v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))], p=0.1),
    ])

    # 2. Instantiate the Dataset (Preprocessing / file searching happens here)
    dataset = Sam3ReIDDataset(
        root_dir="/z/dat/person_reid/internal/input_videos",
        transform=None,
        target_size=(1080, 1920)
    )

    # 3. Setup PK parameters
    P = 16 # Number of unique people per batch
    K = 4  # Number of frames per person
    # Total Batch Size = 16 * 4 = 64

    # 4. Instantiate the custom Sampler
    sampler = VideoSlicePKBatchSampler(
        dataset=dataset, 
        num_identities=P, 
        instances_per_identity=K
    )

    # 5. Create the Dataloader
    # Note: reid_collate_fn handles fallback if PIL images/arrays cannot be default stacked.
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=8,
        pin_memory=True,
        # collate_fn=reid_collate_fn
    )

    dino_dataloader = DinoDataLoaderWrapper(dataloader, transform=transform)

    # Iterate
    for batch_idx, batch in enumerate(dino_dataloader):
        images = batch["image"]
        labels = batch["class_id"]
        masks = batch["mask"]
        embeddings = batch["embeddings"]

        # Compute the average length of the embeddings and the std of the
        embedding_lengths = [len(emb) for emb in embeddings]
        print(f"Average embedding length: {np.mean(embedding_lengths)}")
        print(f"Std of embedding lengths: {np.std(embedding_lengths)}")
        
        # Pass to model and Circle Loss...
        print(f"Images length: {len(images)}, First image shape: {images[0].shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Masks length: {len(masks)}, First mask shape: {masks[0].shape}")
        break

    # Sample batches as fast as possible to test the speed
    import time
    from tqdm import tqdm
    sample_test_len_s = 300
    start_time = time.time()
    with Timer("Total", text="Total: {:.4f} seconds"):
        for batch_idx, batch in enumerate(tqdm(dino_dataloader)):
            if time.time() - start_time > sample_test_len_s:
                break
            
    num_batches = batch_idx + 1
    duration = time.time() - start_time
    print(f"Sampled {num_batches} batches in {duration:.2f} seconds. {num_batches / duration:.2f} batches/second")

    print(Timer.timers)
