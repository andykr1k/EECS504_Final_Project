# TrackAndMask

TrackAndMask is a research codebase for consistent multi-person video de-identification. The project builds a person Re-ID dataset from long-form videos, learns identity embeddings from SAM3 segmentations and DINO features, and uses those embeddings to cluster people consistently across a video.

## Overview

The repository supports the full workflow:

1. Split long videos into shorter scene-consistent clips.
2. Run SAM3 over those clips to create person segmentations and sidecar metadata.
3. Optionally convert external Re-ID datasets such as PRW into the same on-disk format.
4. Inspect the resulting dataset and segmentations with Streamlit tools.
5. Train a person Re-ID model on segmentation-aligned DINO features.
6. Run a video-level Re-ID pipeline that forms tracklets, clusters them, and renders annotated output videos.

## Repository Layout

```text
data/
  process_prw.py
  sam3_reid_dataset.py

models/
  dino_lib.py

pipelines/
  preprocess_videos.py
  video_person_segmenter.py
  video_reid.py

training/
  train.py

analysis/
  sam3_reid_dataset_viewer.py
  view_segmentations.py

maintenance/
  reset_data.py
```

What lives where:

- `data/` contains dataset construction and loading code.
- `models/` contains reusable model-side utilities, most notably the DINO feature extraction harness.
- `pipelines/` contains end-to-end video processing scripts.
- `training/` contains the embedding model definitions, losses, and training loop.
- `analysis/` contains interactive inspection tools built with Streamlit.
- `maintenance/` contains cleanup utilities for generated data.

## Setup

Create an environment and install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Many scripts assume the repository root is on `PYTHONPATH`, and several of the reorganized modules still use direct intra-repo imports. A safe way to launch the project from the repo root is:

```bash
export PYTHONPATH=.:data:models:training
```

Notes:

- The heavy segmentation and Re-ID pipelines are designed around GPU execution and are much slower on CPU.
- Some scripts use additional scientific Python packages beyond the minimal `requirements.txt` list, especially for analysis and training utilities.
- A few pipeline entrypoints still use example hardcoded paths in their `__main__` blocks, so importing the functions directly is often the cleanest way to run custom jobs.

## Dataset Format

The training data is stored as directories ending in `_segmentations`, and the dataset loader searches for them recursively under a root directory.

```text
<dataset_root>/
  <group_name>/
    <video_name>_segmentations/
      <slice_name>/
        <slice_name>_<frame_idx>_frame.png
        <slice_name>_<frame_idx>_segmentation.png
        <slice_name>_<frame_idx>_sidecar.json
```

File semantics:

- `*_frame.png`: the RGB frame saved for training or inspection.
- `*_segmentation.png`: an 8-bit segmentation mask.
- `*_sidecar.json`: metadata about the frame and visible IDs.

The segmentation convention currently used in the code is:

- background: `255`
- foreground person instances: integer object IDs assigned within a slice

The sidecar format is:

```json
{
  "video_path": "path/to/source/video.mp4",
  "frame_idx": 42,
  "visible_person_ids": [0, 1, 4]
}
```

The loader in `data/sam3_reid_dataset.py` uses:

- the parent folder name as a coarse group label
- the `_segmentations` folder name as the video identity
- the slice folder name as the local temporal unit for PK sampling

## Typical Workflow

### 1. Preprocess raw videos into scene slices

`pipelines/preprocess_videos.py` uses TransNetV2 to find scene boundaries and writes shorter clips into `<video_stem>_processed/`.

Programmatic usage:

```python
from pathlib import Path
from pipelines.preprocess_videos import main

main(Path("/path/to/input_videos"), threshold=0.2, max_workers=4)
```

The script's `__main__` block contains example train/val roots that were used during development.

### 2. Generate SAM3 person segmentations

`pipelines/video_person_segmenter.py` reads preprocessed clips, runs SAM3 video segmentation, and writes the dataset format described above.

Programmatic usage:

```python
from pathlib import Path
from pipelines.video_person_segmenter import main

main(Path("/path/to/input_videos"))
```

This script expects each video directory to contain a `config.yaml` describing whether the dataset is enabled and how aggressively frames should be sampled and saved.

Example `config.yaml`:

```yaml
enabled: true
sam_prompt: "Child, Children, Student, Students"
sam_skip_frames: 1
save_skip_frames: 5
max_scene_len_frames: 600
```

### 3. Convert PRW into the same dataset structure

`data/process_prw.py` converts PRW frames and annotations into per-identity `_segmentations` folders so they can be loaded by the same dataset code.

Programmatic usage:

```python
from pathlib import Path
from data.process_prw import preprocess_prw_for_sam3_reid

preprocess_prw_for_sam3_reid(
    Path("/path/to/prw_root"),
    Path("/path/to/output_dir"),
)
```

### 4. Inspect generated data

There are two Streamlit apps in `analysis/`:

- `analysis/view_segmentations.py`: browse segmentation folders frame by frame
- `analysis/sam3_reid_dataset_viewer.py`: sample PK batches and inspect DINO-aligned training batches

Example commands:

```bash
PYTHONPATH=.:data:models:training streamlit run analysis/view_segmentations.py
PYTHONPATH=.:data:models:training streamlit run analysis/sam3_reid_dataset_viewer.py
```

### 5. Train the Re-ID model

`training/train.py` contains:

- circle loss
- two model variants: an MLP-based head and a transformer-based head
- PK sampling and validation logic
- optional UMAP/PCA logging and Weights & Biases integration

Example training command:

```bash
PYTHONPATH=.:data:models:training python training/train.py \
  --train_dir /z/dat/person_reid/train \
  --val_dir /z/dat/person_reid/val \
  --model_type transformer
```

Larger DINO backbone example:

```bash
PYTHONPATH=.:data:models:training python training/train.py \
  --train_dir /z/dat/person_reid/train \
  --val_dir /z/dat/person_reid/val \
  --model_type transformer \
  --dino_checkpoint facebook/dinov3-vitl16-pretrain-lvd1689m \
  --dino_dim 1024 \
  --train_max_batches 500
```

### 6. Run video-level Re-ID and rendering

`pipelines/video_reid.py` is the end-to-end inference pipeline. It:

1. splits a video into scenes
2. runs SAM3 segmentation
3. aligns masks with DINO patch embeddings
4. applies the learned contrastive head
5. forms tracklets
6. clusters tracklets into global person IDs
7. renders an annotated output video

The `__main__` block currently contains a concrete example configuration with hardcoded paths. For custom runs, the cleanest interface is to import:

- `ReIDConfig`
- `SAMHarness`
- `run_video_reid_pipeline`

from `pipelines/video_reid.py`, then instantiate the trained head you want to use.

## Cleanup Utilities

`maintenance/reset_data.py` can remove generated `_processed` and `_segmentations` directories when you want to rerun dataset generation from scratch. The default `__main__` block shows one example target directory, and the main entrypoint is:

```python
from pathlib import Path
from maintenance.reset_data import main

main(
    Path("/path/to/dataset_root"),
    remove_segmentations=True,
    remove_preprocessed_slices=False,
    use_parallel=True,
    dry_run=False,
)
```

## Important Files

- `data/sam3_reid_dataset.py`: dataset loader, custom sampler, augmentation helpers, and DINO batch wrapper
- `models/dino_lib.py`: DINO preprocessing and segmentation-to-patch alignment
- `training/train.py`: Re-ID model definitions and training loop
- `pipelines/preprocess_videos.py`: scene slicing and clip export
- `pipelines/video_person_segmenter.py`: SAM3-based dataset generation
- `pipelines/video_reid.py`: full inference, tracklet formation, clustering, and rendering
- `analysis/view_segmentations.py`: frame-level segmentation browser
- `analysis/sam3_reid_dataset_viewer.py`: PK batch visualizer
- `maintenance/reset_data.py`: cleanup tool for generated `_processed` and `_segmentations` directories

## Project Context

This repository was developed as an EECS 504 final project. The original course proposal is in `Proposal.md`.
