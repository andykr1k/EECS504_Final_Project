import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from typing import Tuple
import math
import numpy as np
from dataclasses import dataclass
from codetiming import Timer
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import concurrent.futures


BboxEmbeddingMap = dict[Tuple[int, int, int, int], torch.Tensor]

@dataclass
class DinoSegmentation:
    person_id: int
    dino_embeddings: list[torch.Tensor]
    dino_overlaps: list[float]
    dino_bboxes: list[tuple[int, int, int, int]]

@dataclass
class OptimizedDinoSegmentation:
    person_id: int
    dino_embeddings: torch.Tensor
    dino_overlaps: torch.Tensor
    dino_bboxes: torch.Tensor

class DinoHarness:
    def __init__(self, checkpoint="facebook/dinov3-vits16-pretrain-lvd1689m", device="cuda", max_side_len=1024):
        self.device = device
        self.max_side_len = max_side_len

        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).to(device)
        self.model.eval()

        if hasattr(self.model.config, "patch_size"):
            self.patch_size = self.model.config.patch_size
            assert (self.max_side_len / self.patch_size).is_integer(), "max_side_len must be multiple of patch_size"
        else:
            raise ValueError("Patch size not found in model config")

    @Timer(name="Resize and Pad", text="Resize and Pad: {:.4f} seconds")
    def resize_and_pad(self, imgs: list[Image.Image]) -> Tuple[list[Image.Image], list[Tuple[int, int]]]:
        """
        1. Resizes image so longest side is max_side_len.
        2. Pads onto a square black canvas.
        
        Returns:
            padded_images: list[Image.Image] (All square and same size)
            grid_sizes: list[Tuple(h_patches, w_patches)] - The dimensions of the valid content in patches
        """
        processed_images = []
        grid_sizes = []

        for img in imgs:
            w, h = img.size
            
            # Scale the longest side to the max side length
            scale = self.max_side_len / max(w, h)
            new_w = w * scale
            new_h = h * scale

            # Snap dimensions to patch multiples
            resize_w = int(round(new_w / self.patch_size)) * self.patch_size
            resize_h = int(round(new_h / self.patch_size)) * self.patch_size

            resized_img = img.resize((resize_w, resize_h), resample=Image.BICUBIC)

            # Paste onto square canvas
            padded_img = Image.new("RGB", (self.max_side_len, self.max_side_len), (0, 0, 0))
            padded_img.paste(resized_img, (0, 0))
            processed_images.append(padded_img)

            # Calculate valid grid dimensions
            valid_grid_h = resize_h // self.patch_size
            valid_grid_w = resize_w // self.patch_size
            grid_sizes.append((valid_grid_h, valid_grid_w))
            
        return processed_images, grid_sizes

    def extract_patch_features(self, imgs: list[Image.Image]) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Extracts features and slices off the padding.
        
        Returns:
            features_list: List[torch.Tensor]
                           A list where each element is a tensor of shape (Valid_Grid_H, Valid_Grid_W, Dim).
                           Shapes will vary between images in the batch depending on aspect ratio.
            cls_tokens: Tensor (Batch, Dim)
        """
        # Pad to squares of the same size for the model
        padded_images, grid_sizes = self.resize_and_pad(imgs)

        with Timer(name="Extract Patch Features", text="Extract Patch Features: {:.4f} seconds"):

            with Timer(name="Process Images for DINO", text="Process Images for DINO: {:.4f} seconds"):
                inputs = self.processor(
                    images=padded_images,
                    do_resize=False,
                    do_center_crop=False,
                    do_pad=False,
                    return_tensors="pt"
                ).to(self.device)

            with torch.no_grad():
                with Timer(name="Run DINO Model", text="Run DINO Model: {:.4f} seconds"):
                    outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state

                # DINOv3: [CLS, REG*4, PATCHES...]
                cls_tokens = last_hidden_state[:, 0, :]
                patch_tokens = last_hidden_state[:, 5:, :]

                # Reshape patches to full square grid first
                grid_dim = self.max_side_len // self.patch_size
                b, n_patches, dim = patch_tokens.shape

                # Safety check
                if n_patches != grid_dim * grid_dim:
                    grid_dim = int(math.sqrt(n_patches))

                # (Batch, Max_H, Max_W, Dim)
                full_grid_embeddings = patch_tokens.reshape(b, grid_dim, grid_dim, dim)
            
            # Slice off padding based on grid_sizes
            features_list = []
            full_grid_embeddings = full_grid_embeddings.detach()
            
            for i, (valid_h, valid_w) in enumerate(grid_sizes):
                valid_features = full_grid_embeddings[i, :valid_h, :valid_w, :].clone()
                features_list.append(valid_features)
            
            return features_list, cls_tokens
            
    def get_bbox_to_embedding_map(self, images: list[Image.Image]) -> list[BboxEmbeddingMap]:
        """
        Returns a mapping from pixel bounding boxes in the original image to the corresponding DINOv3 embeddings.

        Returns:
            List of functions that map (x1, y1, x2, y2) to a list of embeddings.
        """
        features_list, _ = self.extract_patch_features(images)

        with Timer("Build Bbox to Embedding Map", text="Build Bbox to Embedding Map: {:.4f} seconds"):
            results = []
            for i, img in enumerate(images):
                orig_w, orig_h = img.size
                embeddings = features_list[i]  # (Grid_H, Grid_W, Dim)
                grid_h, grid_w, _ = embeddings.shape

                # Calculate the size of the resized image
                valid_pixel_w = grid_w * self.patch_size
                valid_pixel_h = grid_h * self.patch_size

                # Calculate the scale factor
                scale_x = valid_pixel_w / orig_w
                scale_y = valid_pixel_h / orig_h

                bbox_to_embedding = {}
                for gy in range(grid_h):
                    for gx in range(grid_w):
                        vx1 = gx * self.patch_size
                        vy1 = gy * self.patch_size
                        vx2 = (gx + 1) * self.patch_size
                        vy2 = (gy + 1) * self.patch_size

                        px1 = int(round(vx1 / scale_x))
                        py1 = int(round(vy1 / scale_y))
                        px2 = int(round(vx2 / scale_x))
                        py2 = int(round(vy2 / scale_y))

                        bbox_to_embedding[(px1, py1, px2, py2)] = embeddings[gy, gx]
                
                results.append(bbox_to_embedding)
        
        return results

    def match_segmentations_to_dino(self, images: list[Image.Image], segs: list[np.ndarray] | list[torch.Tensor]):
        """
        For each image, and for each person in the segmentation, find the best matching DINOv3 patch embedding.

        each segmentation is an array of shape (H, W) where each value is the ID of the person in that pixel.

        Returns:
            List of lists of embeddings, where each inner list corresponds to the people in the image.
        """
        dino_segmentations: list[list[DinoSegmentation]] = []
        bbox_to_embedding_maps = self.get_bbox_to_embedding_map(images)

        with Timer("Match Segmentations to DINO", text="Match Segmentations to DINO: {:.4f} seconds"):
            for img_index in range(len(images)):
                image_dino_segmentations: list[DinoSegmentation] = []

                segmentation = segs[img_index]
                if isinstance(segmentation, torch.Tensor):
                    segmentation = segmentation.cpu().numpy()
            
                assert images[img_index].size == segmentation.shape[::-1], f"Image {img_index} size {images[img_index].size} does not match segmentation size {segs[img_index].shape}"

                bbox_to_embedding = bbox_to_embedding_maps[img_index]
                unique_person_ids = np.unique(segmentation)
                for person_id in unique_person_ids:
                    if person_id == -1:
                        continue

                    person_mask = segmentation == person_id

                    dino_bboxes: list[tuple[int, int, int, int]] = []
                    dino_embeddings: list[torch.Tensor] = []
                    dino_overlaps: list[float] = []

                    for bbox, embedding in bbox_to_embedding.items():
                        x1, y1, x2, y2 = bbox
                        mask_within_bbox = person_mask[y1:y2, x1:x2]
                        overlap = np.sum(mask_within_bbox) / (mask_within_bbox.size + 1e-6)
                        if overlap > 0.0:
                            dino_bboxes.append(bbox)
                            dino_embeddings.append(embedding)
                            dino_overlaps.append(overlap)

                    assert len(dino_embeddings) > 0, f"No DINO embeddings found for person {person_id} in image {img_index}"
                    dino_segmentation = DinoSegmentation(person_id, dino_embeddings, dino_overlaps, dino_bboxes)
                    image_dino_segmentations.append(dino_segmentation)
                dino_segmentations.append(image_dino_segmentations)
            return dino_segmentations



class OptimizedDinoHarness:
    def __init__(self, checkpoint="facebook/dinov3-vits16-pretrain-lvd1689m", device="cuda", max_side_len=1024):
        self.device = torch.device(device)
        self.max_side_len = max_side_len

        # Bypass the slow processor, only load the model
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device)
        self.model.eval()

        if hasattr(self.model.config, "patch_size"):
            self.patch_size = self.model.config.patch_size
            assert (self.max_side_len / self.patch_size).is_integer(), "max_side_len must be multiple of patch_size"
        else:
            raise ValueError("Patch size not found in model config")

        # Standard ImageNet normalization used by DINO
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    @Timer(name="Process and Pad Tensors", text="Process and Pad Tensors: {:.4f} seconds")
    def preprocess_images(self, imgs: list[Image.Image] | list[torch.Tensor]) -> tuple[torch.Tensor, list[tuple[int, int]], list[tuple[int, int]]]:
        """
        Batched resizing, padding, and normalization entirely on GPU.
        Returns:
            batched_padded_images: (B, 3, max_side_len, max_side_len)
            grid_sizes: list of (valid_grid_h, valid_grid_w)
            original_sizes: list of (orig_w, orig_h)
        """
        processed_tensors = []
        grid_sizes = []
        original_sizes = []

        for img in imgs:
            # 1. Ensure we have a GPU float tensor (C, H, W) scaled 0-1
            if isinstance(img, Image.Image):
                orig_w, orig_h = img.size
                img_t = TF.to_tensor(img).to(self.device)
            else:
                orig_h, orig_w = img.shape[1], img.shape[2]
                img_t = img.to(self.device)
                if img_t.dtype != torch.float32 and img_t.dtype != torch.float16:
                    img_t = img_t.float() / 255.0

            original_sizes.append((orig_w, orig_h))

            # 2. Calculate scales
            scale = self.max_side_len / max(orig_w, orig_h)
            new_w, new_h = orig_w * scale, orig_h * scale

            resize_w = int(round(new_w / self.patch_size)) * self.patch_size
            resize_h = int(round(new_h / self.patch_size)) * self.patch_size

            grid_sizes.append((resize_h // self.patch_size, resize_w // self.patch_size))

            # 3. Resize
            # Add batch dim, resize, remove batch dim
            img_resized = F.interpolate(
                img_t.unsqueeze(0), 
                size=(resize_h, resize_w), 
                mode='bicubic', 
                antialias=True
            ).squeeze(0)

            # 4. Pad to max_side_len (pad takes left, right, top, bottom)
            pad_w = self.max_side_len - resize_w
            pad_h = self.max_side_len - resize_h
            img_padded = F.pad(img_resized, (0, pad_w, 0, pad_h), value=0.0)
            
            processed_tensors.append(img_padded)

        # Stack into a single batch
        batched_tensors = torch.stack(processed_tensors)
        
        # 5. Normalize the entire batch at once
        batched_tensors = (batched_tensors - self.mean) / self.std

        return batched_tensors, grid_sizes, original_sizes

    def extract_patch_features(self, imgs: list[Image.Image] | list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor, list[tuple[int, int]], list[tuple[int, int]]]:
        batched_images, grid_sizes, original_sizes = self.preprocess_images(imgs)

        with torch.no_grad():
            with Timer(name="Run DINO Model", text="Run DINO Model: {:.4f} seconds"):
                outputs = self.model(pixel_values=batched_images)
            
            last_hidden_state = outputs.last_hidden_state
            cls_tokens = last_hidden_state[:, 0, :]
            patch_tokens = last_hidden_state[:, 5:, :]

            grid_dim = self.max_side_len // self.patch_size
            b, n_patches, dim = patch_tokens.shape

            if n_patches != grid_dim * grid_dim:
                grid_dim = int(math.sqrt(n_patches))

            full_grid_embeddings = patch_tokens.reshape(b, grid_dim, grid_dim, dim)
        
        features_list = []
        for i, (valid_h, valid_w) in enumerate(grid_sizes):
            features_list.append(full_grid_embeddings[i, :valid_h, :valid_w, :].clone())
        
        return features_list, cls_tokens, grid_sizes, original_sizes

    def match_segmentations_to_dino(
        self, 
        images: list[torch.Tensor] | list[torch.Tensor],  
        segs: list[np.ndarray] | list[torch.Tensor]
    ) -> list[list[OptimizedDinoSegmentation]]:
        """
        Uses GPU-accelerated Average Pooling to instantly map masks to patch embeddings.
        """
        features_list, _, grid_sizes, original_sizes = self.extract_patch_features(images)

        dino_segmentations = []

        with Timer("Match Segmentations to DINO (Batched)", text="Match Segmentations: {:.4f} seconds"):
            for img_idx in range(len(images)):
                image_dino_segmentations = []
                segmentation = segs[img_idx]
                
                # Ensure segmentation is a tensor on the correct device
                if isinstance(segmentation, np.ndarray):
                    segmentation = torch.from_numpy(segmentation)
                segmentation = segmentation.to(self.device)

                embeddings = features_list[img_idx]  # (Grid_H, Grid_W, Dim)
                valid_grid_h, valid_grid_w = grid_sizes[img_idx]
                orig_w, orig_h = original_sizes[img_idx]

                valid_pixel_w = valid_grid_w * self.patch_size
                valid_pixel_h = valid_grid_h * self.patch_size
                scale_x = valid_pixel_w / orig_w
                scale_y = valid_pixel_h / orig_h

                unique_person_ids = torch.unique(segmentation)

                for person_id in unique_person_ids:
                    if person_id == -1:
                        continue

                    # Create binary mask (1 for person, 0 for background)
                    with Timer("Create Binary Mask", text="Create Binary Mask: {:.4f} seconds"):
                        person_mask = (segmentation == person_id).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

                    # Resize mask to the valid image space (nearest neighbor to keep 0s and 1s)
                    with Timer("Resize Mask", text="Resize Mask: {:.4f} seconds"):
                        mask_resized = F.interpolate(person_mask, size=(valid_pixel_h, valid_pixel_w), mode='nearest')

                    # ---------------------------------------------------------
                    # THE MAGIC: AvgPool2D acts as a sliding patch window.
                    # It instantly calculates the exact % overlap for every patch!
                    # ---------------------------------------------------------
                    with Timer("Calculate Overlaps", text="Calculate Overlaps: {:.4f} seconds"):
                        overlaps = F.avg_pool2d(mask_resized, kernel_size=self.patch_size, stride=self.patch_size).squeeze() # (Grid_H, Grid_W)

                    # Find patches where overlap > 0
                    with Timer("Find Valid Patches", text="Find Valid Patches: {:.4f} seconds"):
                        valid_y, valid_x = torch.where(overlaps > 0.0)

                    if len(valid_y) == 0:
                        continue

                    # Vectorized bounding box calculation
                    with Timer("Calculate Bounding Boxes", text="Calculate Bounding Boxes: {:.4f} seconds"):
                        px1 = torch.round((valid_x * self.patch_size) / scale_x).int()
                        py1 = torch.round((valid_y * self.patch_size) / scale_y).int()
                        px2 = torch.round(((valid_x + 1) * self.patch_size) / scale_x).int()
                        py2 = torch.round(((valid_y + 1) * self.patch_size) / scale_y).int()

                        dino_bboxes = torch.stack([px1, py1, px2, py2], dim=1)
                    dino_overlaps = overlaps[valid_y, valid_x]
                    
                    # Extract embeddings for the valid patches
                    dino_embeddings = embeddings[valid_y, valid_x]

                    image_dino_segmentations.append(
                        OptimizedDinoSegmentation(person_id.item(), dino_embeddings, dino_overlaps, dino_bboxes)
                    )
                
                dino_segmentations.append(image_dino_segmentations)
                
        return dino_segmentations

    def match_segmentations_to_dino_bool(
        self, 
        images: list[torch.Tensor] | list[torch.Tensor],  
        segs: list[np.ndarray] | list[torch.Tensor]
    ) -> list[list[OptimizedDinoSegmentation]]:
        
        features_list, _, grid_sizes, original_sizes = self.extract_patch_features(images)
        dino_segmentations = []

        with Timer("Match Segmentations to DINO (Algorithmic Reduction)", text="Match Segmentations: {:.4f} seconds"):
            for img_idx in range(len(images)):
                segmentation = segs[img_idx]
                
                if isinstance(segmentation, np.ndarray):
                    segmentation = torch.from_numpy(segmentation)
                
                # Move to device and ensure it's a float for interpolation
                segmentation = segmentation.to(self.device, non_blocking=True).float()

                valid_grid_h, valid_grid_w = grid_sizes[img_idx]
                orig_w, orig_h = original_sizes[img_idx]

                person_mask = segmentation.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

                # ---------------------------------------------------------
                # THE OPTIMIZATION: One step. Direct to grid size.
                # 'area' interpolation on a 0/1 mask calculates exact overlap %.
                # ---------------------------------------------------------
                overlaps = F.interpolate(
                    person_mask, 
                    size=(valid_grid_h, valid_grid_w), 
                    mode='area'
                ).squeeze() # (grid_h, grid_w)

                valid_y, valid_x = torch.where(overlaps > 0.0)

                if len(valid_y) == 0:
                    dino_segmentations.append([])
                    continue

                # Vectorized bounding box calculation
                valid_pixel_w = valid_grid_w * self.patch_size
                valid_pixel_h = valid_grid_h * self.patch_size
                scale_x = valid_pixel_w / orig_w
                scale_y = valid_pixel_h / orig_h

                px1 = torch.round((valid_x * self.patch_size) / scale_x).int()
                py1 = torch.round((valid_y * self.patch_size) / scale_y).int()
                px2 = torch.round(((valid_x + 1) * self.patch_size) / scale_x).int()
                py2 = torch.round(((valid_y + 1) * self.patch_size) / scale_y).int()

                dino_bboxes = torch.stack([px1, py1, px2, py2], dim=1)
                dino_overlaps = overlaps[valid_y, valid_x]
                dino_embeddings = features_list[img_idx][valid_y, valid_x]

                dino_segmentations.append([
                    OptimizedDinoSegmentation(1, dino_embeddings, dino_overlaps, dino_bboxes)
                ])
                
        return dino_segmentations

if __name__ == "__main__":
    import time
    
    print("Setting up benchmark...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create mock batch of 8 high-res images and masks
    batch_size = 64
    mock_images = [torch.rand(3, 720, 1280) for _ in range(batch_size)]
    
    # Create mock masks with 3 distinct "people" (ids 0, 1, 2) and background (-1)
    mock_masks = []
    for _ in range(batch_size):
        mask = torch.full((720, 1280), -1, dtype=torch.long)
        mask[100:300, 200:400] = 0  # Person 0
        mask[400:600, 800:1000] = 1 # Person 1
        mask[200:500, 500:700] = 2  # Person 2
        mock_masks.append(mask)

    PIL_mock_images = [TF.to_pil_image(img) for img in mock_images]

    print("Initializing Models (This takes a moment)...")
    harness = OptimizedDinoHarness(device=device)
    # harness = DinoHarness(device=device)
    
    # You can import your original DinoHarness here if you want to run it side-by-side.
    # from original_harness import DinoHarness
    # old_harness = DinoHarness(device=device)

    print("\n--- Running Optimized Harness ---")
    start = time.perf_counter()
    new_results = harness.match_segmentations_to_dino(mock_images, mock_masks)
    # new_results = harness.match_segmentations_to_dino(PIL_mock_images, mock_masks)
    torch.cuda.synchronize() if device == "cuda" else None
    new_time = time.perf_counter() - start
    
    print(f"Optimized Pipeline Time: {new_time:.4f} seconds")
    print(f"Total people detected across {batch_size} images: {sum(len(img) for img in new_results)}")
    
    # Note for DataLoader Wrapper:
    # Update your DinoDataLoaderWrapper to yield the raw Tensors instead of F.to_pil_image.
    # e.g., Replace: images = [F.to_pil_image(img) for img in batch["images"]]
    #       With:    images = batch["images"]

    print(Timer.timers)