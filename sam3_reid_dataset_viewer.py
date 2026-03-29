import streamlit as st
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import your updated components
# (Ensure the import paths match where your classes actually live)
from sam3_reid_dataset_v2 import (
    Sam3ReIDDataset, 
    VideoSlicePKBatchSampler, 
    reid_collate_fn, 
    DinoDataLoaderWrapper
)

st.set_page_config(page_title="Re-ID Dataset Viewer", layout="wide")

st.title("🏃‍♂️ Person Re-ID Batch Visualizer")
st.markdown("This app samples a single $P \\times K$ batch from the dataset to verify that identities are grouped correctly, DINO embeddings are mapped, and background masking is working.")

# --- Sidebar Configuration ---
st.sidebar.header("Dataset Configuration")
data_dir = st.sidebar.text_input("Dataset Directory", value="/z/dat/person_reid/internal/input_videos")

st.sidebar.header("PK Sampler Params")
P = st.sidebar.number_input("P (Identities per batch)", min_value=2, max_value=32, value=4)
K = st.sidebar.number_input("K (Instances per identity)", min_value=2, max_value=16, value=4)

# --- Caching the Dataset ---
@st.cache_resource(show_spinner="Scanning dataset directory...")
def load_dataset(root_dir):
    # We pass transform=None to keep images as standard [0, 1] float tensors 
    # without ImageNet normalization, so they render accurately in Streamlit.
    dataset = Sam3ReIDDataset(
        root_dir=root_dir,
        transform=None 
    )
    return dataset

# --- Main App Logic ---
try:
    dataset = load_dataset(data_dir)
    st.sidebar.success(f"Loaded {len(dataset)} bounding boxes!")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

if st.button("🎲 Sample New Batch", type="primary"):
    with st.spinner("Sampling and calculating DINO embeddings..."):
        # 1. Initialize Sampler (Using updated kwargs)
        sampler = VideoSlicePKBatchSampler(
            dataset, 
            num_identities=P, 
            instances_per_identity=K
        )
        
        # 2. Initialize DataLoader
        # num_workers=0 is necessary for Streamlit to prevent threading issues
        dataloader = DataLoader(
            dataset, 
            batch_sampler=sampler, 
            num_workers=0,
            collate_fn=reid_collate_fn
        )
        
        # 3. Wrap with DINO DataLoader
        # (Assuming DINO is available on CUDA, otherwise you may need device="cpu")
        dino_dataloader = DinoDataLoaderWrapper(dataloader)
        
        # 4. Grab the first batch
        # The new wrapper yields a dictionary rather than a tuple
        batch = next(iter(dino_dataloader))
        
        batch_images = batch["images"]
        batch_labels = batch["class_ids"]
        batch_masks = batch["masks"]
        batch_bboxes = batch["bboxes"]
        batch_embeddings = batch["embeddings"]
        
        st.subheader(f"Batch Size: {P * K} ({P} identities, {K} frames each)")
        st.divider()

        # Render the P x K Grid
        # Each row is an identity (P), each column is an instance (K)
        for p_idx in range(P):
            cols = st.columns(K)
            
            # The sampler guarantees that the batch is strictly ordered by identity
            start_idx = p_idx * K
            current_class_id = batch_labels[start_idx].item() if hasattr(batch_labels[start_idx], 'item') else batch_labels[start_idx]
            
            cols[0].markdown(f"**Class ID: `{current_class_id}`**")

            for k_idx in range(K):
                batch_idx = start_idx + k_idx
                
                # Retrieve individual items
                img_tensor = batch_images[batch_idx]  # Shape: [C, H, W]
                mask_tensor = batch_masks[batch_idx]  # Shape: [1, H, W] or [H, W]
                bboxes = batch_bboxes[batch_idx]
                embeddings = batch_embeddings[batch_idx]
                
                # Convert Image to [H, W, C] numpy float32 array in range [0, 1]
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.float32)
                
                # Squeeze mask to 2D [H, W] boolean array
                mask_np = mask_tensor.squeeze().cpu().numpy().astype(bool)
                
                if len(embeddings) > 0:
                    # Handle both lists of tensors and single stacked tensors safely
                    if isinstance(embeddings, list):
                        embeddings_tensor = torch.stack(embeddings)
                    else:
                        embeddings_tensor = embeddings
                        
                    mean = embeddings_tensor.mean(dim=0, keepdim=True)
                    centered = embeddings_tensor - mean
                    
                    q = min(3, centered.shape[0])
                    if q > 0:
                        # PCA calculation
                        U, S, V = torch.pca_lowrank(centered, q=q)
                        reduced = torch.matmul(centered, V)
                        
                        # Pad with zeros if less than 3 components exist
                        if q < 3:
                            padding = torch.zeros(reduced.shape[0], 3 - q, device=reduced.device)
                            reduced = torch.cat([reduced, padding], dim=-1)
                            
                        # Normalize PCA outputs to [0, 1] for RGB mapping
                        min_vals = reduced.min(dim=0, keepdim=True)[0]
                        max_vals = reduced.max(dim=0, keepdim=True)[0]
                        ranges = max_vals - min_vals
                        ranges[ranges == 0] = 1.0 # prevent division by zero
                        
                        normalized_colors = ((reduced - min_vals) / ranges).cpu().numpy()
                        
                        # Create a floating point overlay
                        overlay = np.zeros_like(img_np)
                        
                        for bbox, color in zip(bboxes, normalized_colors):
                            # Ensure bboxes are integers
                            x1, y1, x2, y2 = map(int, bbox)
                            
                            box_mask = np.zeros_like(mask_np, dtype=bool)
                            box_mask[y1:y2, x1:x2] = True
                            
                            # Intersection of SAM mask and DINO bounding box
                            final_mask = mask_np & box_mask
                            overlay[final_mask] = color
                            
                        # Blend image and overlay using mask (all operations in float [0, 1])
                        alpha = 0.6
                        img_np[mask_np] = (img_np[mask_np] * (1 - alpha)) + (overlay[mask_np] * alpha)
                
                # Clip values just in case blending caused overflow, then render
                img_np = np.clip(img_np, 0.0, 1.0)
                cols[k_idx].image(img_np, use_container_width=True)
            
            st.divider()