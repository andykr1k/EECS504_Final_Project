import streamlit as st
import numpy as np
import cv2
import json
import torch
from pathlib import Path
from PIL import Image

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
@st.cache_resource
def get_dino_harness():
    import dino_lib
    return dino_lib.DinoHarness()

@st.cache_data
def get_segmentation_dirs(base_path: str):
    """Finds all directories ending with '_segmentations'."""
    root = Path(base_path)
    if not root.exists():
        return []
    # rgboing allows finding segmentations nested under project directories
    return sorted([d for d in root.rglob("*_segmentations") if d.is_dir()])

@st.cache_data
def get_slice_dirs(segmentation_dir: Path):
    """Finds all slice subdirectories within a segmentation directory."""
    return sorted([d for d in segmentation_dir.glob("*") if d.is_dir()])

@st.cache_data
def get_available_frames(slice_dir: Path):
    """
    Extracts sorted frame indices from the available _frame.png files.
    Accounts for the fact that some frames might be skipped if no people were detected.
    """
    frame_files = list(slice_dir.glob("*_frame.png"))
    frame_indices = []
    for f in frame_files:
        try:
            # Filename format: <video_slice_stem>_<frame_idx>_frame.png
            idx = int(f.stem.split('_')[-2])
            frame_indices.append(idx)
        except ValueError:
            continue
    return sorted(frame_indices)

def get_color_for_id(person_id: int):
    """Generates a stable color based on the person ID."""
    if person_id == -1:
        return [0, 0, 0] # Background
    np.random.seed(person_id)
    return np.random.randint(0, 255, size=3).tolist()

def overlay_masks(image_path: Path, segmentation_path: Path, alpha: float = 0.5, use_dino: bool = False, dino_harness = None):
    """Loads the image and mask, and blends them together."""
    # Load Image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load Segmentation Mask
    try:
        seg_data = np.load(str(segmentation_path))
        segmentation = seg_data['segmentation']
    except Exception as e:
        st.error(f"Error loading segmentation data: {e}")
        return img
        
    overlay = np.zeros_like(img)
    
    # Get unique person IDs in this frame
    unique_ids = np.unique(segmentation)
    
    # Colorize the overlay
    for uid in unique_ids:
        if uid == -1:  # Skip background
            continue
        color = get_color_for_id(uid)
        overlay[segmentation == uid] = color
        
    if use_dino and dino_harness is not None:
        image_pil = Image.fromarray(img)
        with torch.no_grad():
            d_segs = dino_harness.match_segmentations_to_dino([image_pil], [segmentation])
            
        for seg in d_segs[0]:
            if not seg.dino_embeddings:
                continue
                
            uid = seg.person_id
            embeddings = torch.stack(seg.dino_embeddings)
            
            # Center embeddings
            mean = embeddings.mean(dim=0, keepdim=True)
            centered = embeddings - mean
            
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
                
                for bbox, color in zip(seg.dino_bboxes, colors):
                    x1, y1, x2, y2 = bbox
                    box_mask = np.zeros_like(segmentation, dtype=bool)
                    box_mask[y1:y2, x1:x2] = True
                    final_mask = (segmentation == uid) & box_mask
                    overlay[final_mask] = color.tolist()

    # Create mask of where segmentation exists to only blend those areas
    mask_exists = segmentation != -1
    
    # Blend image and overlay using pure NumPy
    blended = img.copy()
    
    # Multiply by alphas, add them up, and ensure it remains a valid 8-bit image format
    blended[mask_exists] = (
        img[mask_exists] * (1 - alpha) + overlay[mask_exists] * alpha
    ).astype(np.uint8)
    
    return blended

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Dataset Viewer", layout="wide")

st.title("Person Re-ID Dataset Viewer")

# --- Sidebar Controls ---
st.sidebar.header("Dataset Navigation")

# 1. Root Directory Input
default_root = "/z/dat/person_reid/internal/input_videos"
dataset_root = st.sidebar.text_input("Dataset Root Directory", default_root)

segmentation_dirs = get_segmentation_dirs(dataset_root)

if not segmentation_dirs:
    st.warning(f"No '*_segmentations' directories found in `{dataset_root}`. Please check the path.")
    st.stop()

# 2. Select Video / Segmentation Directory
dir_names = [d.name for d in segmentation_dirs]
selected_dir_name = st.sidebar.selectbox("1. Select Top-Level Video", dir_names)
selected_seg_dir = segmentation_dirs[dir_names.index(selected_dir_name)]

# 3. Select Slice
slice_dirs = get_slice_dirs(selected_seg_dir)
if not slice_dirs:
    st.warning("No video slices found in this directory.")
    st.stop()

slice_names = [d.name for d in slice_dirs]
selected_slice_name = st.sidebar.selectbox("2. Select Video Slice", slice_names)
selected_slice_dir = slice_dirs[slice_names.index(selected_slice_name)]

# 4. Select Frame
frame_indices = get_available_frames(selected_slice_dir)
if not frame_indices:
    st.warning("No frames found in this slice.")
    st.stop()

# Use a select_slider so we only land on frames that actually exist in the dataset
selected_frame_idx = st.sidebar.select_slider(
    "3. Select Frame", 
    options=frame_indices,
    value=frame_indices[0]
)

use_dino_pca = st.sidebar.checkbox("Visualize DINO Embeddings (PCA)", value=False)
dino_harness = get_dino_harness() if use_dino_pca else None

# Overlay opacity control
alpha = st.sidebar.slider("Mask Opacity", min_value=0.0, max_value=1.0, value=0.6, step=0.1)

# --- Main Display Area ---
st.subheader(f"Slice: `{selected_slice_name}` | Frame: `{selected_frame_idx}`")

# Construct file paths
stem = selected_slice_name
frame_path = selected_slice_dir / f"{stem}_{selected_frame_idx}_frame.png"
seg_path = selected_slice_dir / f"{stem}_{selected_frame_idx}_segmentation.npz"
sidecar_path = selected_slice_dir / f"{stem}_{selected_frame_idx}_sidecar.json"

# Check if all files exist
if not (frame_path.exists() and seg_path.exists() and sidecar_path.exists()):
    st.error("Missing files for this frame. Expected PNG, NPZ, and JSON files to be present.")
    st.stop()

# Render Image with Masks
col1, col2 = st.columns([2, 1])

with col1:
    blended_img = overlay_masks(frame_path, seg_path, alpha=alpha, use_dino=use_dino_pca, dino_harness=dino_harness)
    st.image(blended_img, caption=f"Frame {selected_frame_idx} with Segmentation Masks", width="stretch")#, use_column_width=True)

with col2:
    st.markdown("### Sidecar Metadata")
    try:
        with open(sidecar_path, 'r') as f:
            sidecar_data = json.load(f)
        st.json(sidecar_data)
        
        # Display a legend for the visible people
        if sidecar_data.get("visible_person_ids"):
            st.markdown("#### Mask Legend")
            for pid in sidecar_data["visible_person_ids"]:
                color = get_color_for_id(pid)
                # Convert RGB to Hex for HTML styling
                hex_color = '#%02x%02x%02x' % tuple(color)
                st.markdown(
                    f"<div style='display: flex; align-items: center; margin-bottom: 4px;'>"
                    f"<div style='width: 20px; height: 20px; background-color: {hex_color}; margin-right: 10px; border-radius: 4px;'></div>"
                    f"Person ID: {pid}"
                    f"</div>", 
                    unsafe_allow_html=True
                )
    except Exception as e:
        st.error(f"Failed to read sidecar file: {e}")