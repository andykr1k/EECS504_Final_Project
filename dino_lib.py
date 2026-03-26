import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from typing import Tuple
import math
import numpy as np
from dataclasses import dataclass

BboxEmbeddingMap = dict[Tuple[int, int, int, int], torch.Tensor]

@dataclass
class DinoSegmentation:
    person_id: int
    dino_embeddings: list[torch.Tensor]
    dino_overlaps: list[float]
    dino_bboxes: list[tuple[int, int, int, int]]

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

        inputs = self.processor(
            images=padded_images,
            do_resize=False,
            do_center_crop=False,
            do_pad=False,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
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

    def match_segmentations_to_dino(self, images: list[Image.Image], segs: list[np.ndarray]):
        """
        For each image, and for each person in the segmentation, find the best matching DINOv3 patch embedding.

        each segmentation is an array of shape (H, W) where each value is the ID of the person in that pixel.

        Returns:
            List of lists of embeddings, where each inner list corresponds to the people in the image.
        """
        dino_segmentations: list[list[DinoSegmentation]] = []
        bbox_to_embedding_maps = self.get_bbox_to_embedding_map(images)
        for img_index in range(len(images)):
            image_dino_segmentations: list[DinoSegmentation] = []

            segmentation = segs[img_index]
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


                

                
            
        