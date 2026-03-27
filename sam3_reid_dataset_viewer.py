import streamlit as st
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
from sam3_reid_dataset import Sam3ReIDDataset, VideoSlicePKBatchSampler, reid_collate_fn, DinoDataLoaderWrapper

st.set_page_config(page_title="Re-ID Dataset Viewer", layout="wide")

st.title("🏃‍♂️ Person Re-ID Batch Visualizer")
st.markdown("This app samples a single $P \\times K$ batch from the dataset to verify that identities are grouped correctly and background masking is working.")

# --- Sidebar Configuration ---
st.sidebar.header("Dataset Configuration")
data_dir = st.sidebar.text_input("Dataset Directory", value="/z/dat/person_reid/internal/input_videos")
mask_bg = st.sidebar.checkbox("Apply SAM Background Mask", value=True)

st.sidebar.header("PK Sampler Params")
P = st.sidebar.number_input("P (Identities per batch)", min_value=2, max_value=32, value=4)
K = st.sidebar.number_input("K (Instances per identity)", min_value=2, max_value=16, value=4)

# --- Caching the Dataset ---
# We cache the dataset initialization so we don't rescan the disk on every UI interaction
@st.cache_resource(show_spinner="Scanning dataset directory...")
def load_dataset(root_dir, mask_background):
    # For visualization, we skip ToTensor() and Normalize() so Streamlit can render PIL images naturally.
    # We remove Resize to keep the full original image dimensions.
    vis_transform = None
    
    dataset = Sam3ReIDDataset(
        root_dir=root_dir,
        transform=vis_transform,
        mask_background=mask_background
    )
    return dataset

# --- Main App Logic ---
try:
    dataset = load_dataset(data_dir, mask_bg)
    st.sidebar.success(f"Loaded {len(dataset)} bounding boxes!")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

if st.button("🎲 Sample New Batch", type="primary"):
    with st.spinner("Sampling..."):
        # Initialize Sampler and Dataloader
        sampler = VideoSlicePKBatchSampler(dataset, num_identities_per_batch=P, instances_per_identity=K)
        
        # num_workers=0 is safer for Streamlit environments
        dataloader = DataLoader(
            dataset, 
            batch_sampler=sampler, 
            num_workers=0,
            collate_fn=reid_collate_fn
        )
        dataloader = DinoDataLoaderWrapper(dataloader)
        
        # Grab the first batch
        batch_images, batch_labels, batch_masks, batch_bboxes, batch_embeddings, batch_overlaps = next(iter(dataloader))
        
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
            
            import numpy as np
            import torch

            for k_idx in range(K):
                batch_idx = start_idx + k_idx
                img = batch_images[batch_idx]
                label = batch_labels[batch_idx]
                mask = batch_masks[batch_idx]
                bboxes = batch_bboxes[batch_idx]
                embeddings = batch_embeddings[batch_idx]
                
                img_np = np.array(img).copy().transpose(1, 2, 0)
                
                if len(embeddings) > 0:
                    embeddings_tensor = torch.stack(embeddings)
                    mean = embeddings_tensor.mean(dim=0, keepdim=True)
                    centered = embeddings_tensor - mean
                    
                    q = min(3, centered.shape[0])
                    if q > 0:
                        U, S, V = torch.pca_lowrank(centered, q=q)
                        reduced = torch.matmul(centered, V)
                        
                        if q < 3:
                            padding = torch.zeros(reduced.shape[0], 3 - q, device=reduced.device)
                            reduced = torch.cat([reduced, padding], dim=-1)
                            
                        min_vals = reduced.min(dim=0, keepdim=True)[0]
                        max_vals = reduced.max(dim=0, keepdim=True)[0]
                        
                        ranges = max_vals - min_vals
                        ranges[ranges == 0] = 1.0 # prevent division by zero
                        
                        normalized = (reduced - min_vals) / ranges
                        colors = (normalized * 255.0).to(torch.uint8).cpu().numpy()
                        
                        overlay = np.zeros_like(img_np)
                        for bbox, color in zip(bboxes, colors):
                            x1, y1, x2, y2 = bbox
                            box_mask = np.zeros_like(mask, dtype=bool)
                            box_mask[y1:y2, x1:x2] = True
                            final_mask = mask & box_mask
                            overlay[final_mask] = color.tolist()
                            
                        # Blend image and overlay using mask
                        alpha = 0.6
                        img_np[mask] = (
                            img_np[mask] * (1 - alpha) + overlay[mask] * alpha
                        ).astype(np.uint8)
                
                # Render image in the respective column
                # normalize image to [0, 1]
                img_np = img_np / 255.0
                cols[k_idx].image(img_np, width="content")
            
            st.divider()