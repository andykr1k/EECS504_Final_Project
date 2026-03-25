import streamlit as st
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
from sam3_reid_dataset import Sam3ReIDDataset, VideoSlicePKBatchSampler

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
    # We only resize to standard Re-ID dimensions.
    vis_transform = T.Compose([
        T.Resize((256, 128))
    ])
    
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
        
        # Custom collate function to handle PIL Images without stacking them into Tensors
        def pil_collate_fn(batch):
            images = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            return images, labels

        # num_workers=0 is safer for Streamlit environments
        dataloader = DataLoader(
            dataset, 
            batch_sampler=sampler, 
            num_workers=0,
            collate_fn=pil_collate_fn
        )
        
        # Grab the first batch
        batch_images, batch_labels = next(iter(dataloader))
        
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
                img = batch_images[batch_idx]
                label = batch_labels[batch_idx]
                
                # Render image in the respective column
                cols[k_idx].image(img, use_container_width=True)
            
            st.divider()