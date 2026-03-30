import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Dict

import yaml
import cv2
import numpy as np
from tqdm import tqdm
import imageio
import transnetv2_pytorch as transnet

@dataclass
class Config:
    enabled: bool
    sam_prompt: str
    sam_skip_frames: int
    save_skip_frames: int
    max_scene_len_frames: int

# ---------------------------------------------------------------------------
# WORKER FUNCTION (Runs in parallel on CPU cores)
# ---------------------------------------------------------------------------
def process_video_worker(input_video_path: Path, out_dir_path: Path, constrained_scenes: List[Dict]):
    """
    This function handles the CPU-heavy decoding, overlay drawing, and encoding.
    It does NOT touch the ML model.
    """
    try:
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            return f"Error: Could not open video {input_video_path}"
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out_dir_path.mkdir(parents=True, exist_ok=True)
        
        debug_video_path = out_dir_path / f"{input_video_path.stem}_debug.mp4"
        debug_writer = imageio.get_writer(
            str(debug_video_path), 
            fps=fps, 
            codec='libx264', 
            pixelformat='yuv420p',
            macro_block_size=None 
        )
        
        red_frame_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        red_frame_rgb[:] = (255, 0, 0) 
        
        current_frame_idx = 0
        scene_index = 0
        chunk_writer: Any = None
        current_scene = constrained_scenes[scene_index]
        
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
            while current_frame_idx > current_scene['end_frame']:
                if chunk_writer is not None:
                    chunk_writer.close()
                    chunk_writer = None
                    
                for _ in range(2): 
                    debug_writer.append_data(red_frame_rgb)
                    
                scene_index += 1
                if scene_index < len(constrained_scenes):
                    current_scene = constrained_scenes[scene_index]
                else:
                    break
                    
            if scene_index >= len(constrained_scenes):
                break
                
            if current_frame_idx < current_scene['start_frame']:
                current_frame_idx += 1
                continue
                
            if chunk_writer is None:
                chunk_filename = out_dir_path / f"{input_video_path.stem}_{scene_index:04d}.mp4"
                chunk_writer = imageio.get_writer(
                    str(chunk_filename), 
                    fps=fps, 
                    codec='libx264', 
                    pixelformat='yuv420p',
                    macro_block_size=None
                )
                
            chunk_writer.append_data(frame_rgb)
            
            # Debug Video Overlay
            debug_frame_bgr = frame_bgr.copy()
            cur_in_scene = current_frame_idx - current_scene['start_frame']
            
            metadata_text = [
                f"Shot ID: {current_scene['shot_id']}",
                f"Frames: {current_scene['start_frame']} - {current_scene['end_frame']}",
                f"Time: {current_scene['start_time']:.3f}s - {current_scene['end_time']:.3f}s",
                f"Current Frame in Scene: {cur_in_scene}",
                f"Probability: {current_scene['probability']:.4f}"
            ]
            
            y_offset, line_spacing = 35, 35
            for i, text in enumerate(metadata_text):
                y = y_offset + (i * line_spacing)
                cv2.putText(debug_frame_bgr, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(debug_frame_bgr, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
            debug_frame_rgb = cv2.cvtColor(debug_frame_bgr, cv2.COLOR_BGR2RGB)
            debug_writer.append_data(debug_frame_rgb)
            
            current_frame_idx += 1
            
        cap.release()
        debug_writer.close()
        if chunk_writer is not None:
            chunk_writer.close()
            
        return f"Success: {input_video_path.stem}"
        
    except Exception as e:
        return f"Error processing {input_video_path.stem}: {str(e)}"

# ---------------------------------------------------------------------------
# MAIN PROCESS (GPU Inference & Task Delegation)
# ---------------------------------------------------------------------------
def main(parent_dir: Path, threshold: float = 0.2, max_workers: int = 4):
    # 1. Initialize Model on the main thread
    print("Loading TransNetV2 to GPU...")
    model = transnet.TransNetV2()
    model = model.to("cuda")
    model.eval()
    
    # 2. Gather all videos to process
    tasks_to_run = []
    
    for dir_path in parent_dir.glob("*"):
        if not dir_path.is_dir():
            continue
            
        config_path = dir_path / "config.yaml"
        if not config_path.exists():
            continue
            
        config = Config(**yaml.safe_load(config_path.read_text()))
        if not config.enabled:
            print(f"Skipping {dir_path.name} (disabled)")
            continue

        for video_path in dir_path.glob("*.mp4"):
            out_dir_path = video_path.parent / (video_path.stem + "_processed")
            if out_dir_path.exists():
                print(f"Skipping {video_path.name} (already processed)")
                continue
            
            tasks_to_run.append((config, video_path, out_dir_path))

    if not tasks_to_run:
        print("No videos to process.")
        return

    # 3. Setup Process Pool
    # We use 'spawn' to prevent CUDA initialization crashes when forking child processes
    mp_context = mp.get_context('spawn')
    cpu_count = os.cpu_count()
    if cpu_count is None:
        print("Could not determine CPU count, defaulting to 1")
        cpu_count = 1
    max_workers = min(max(1, cpu_count - 2), max_workers) # Leave a couple cores for the OS/Main thread
    
    print(f"Starting pipeline with {max_workers} worker processes...")
    
    # We will store future objects here to track completion
    futures = []

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        
        # 4. Run inference sequentially on main thread, push processing to background
        for config, video_path, out_dir_path in tasks_to_run:
            print(f"\nRunning inference on {video_path.name}...")
            scenes = model.detect_scenes(str(video_path), threshold=threshold)
            
            # Quickly grab FPS to calculate times (opening a video once takes a fraction of a second)
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            constrained_scenes = []
            for scene in scenes:
                start_frame = scene['start_frame']
                end_frame = scene['end_frame']
                
                current_start = start_frame
                while current_start <= end_frame:
                    current_end = min(current_start + config.max_scene_len_frames - 1, end_frame)
                    
                    constrained_scenes.append({
                        'shot_id': scene['shot_id'],
                        'start_frame': current_start,
                        'end_frame': current_end,
                        'start_time': current_start / fps if fps else 0.0,
                        'end_time': current_end / fps if fps else 0.0,
                        'probability': scene['probability']
                    })
                    current_start = current_end + 1

            if not constrained_scenes:
                print(f"No scenes found in {video_path.name}.")
                continue
                
            print(f"TransNet found {len(constrained_scenes)} chunks. Dispatching to worker pool...")
            
            # Hand the IO heavy lifting off to a background process
            future = executor.submit(process_video_worker, video_path, out_dir_path, constrained_scenes)
            futures.append(future)

        # 5. Track completion of the background tasks
        print("\nAll videos processed by model. Waiting for disk I/O to finish...")
        with tqdm(total=len(futures), desc="Writing Videos to Disk") as pbar:
            for future in as_completed(futures):
                result = future.result() # This will raise any exceptions caught in the worker
                pbar.set_postfix_str(result[:40]) # Show the result message briefly
                pbar.update(1)

if __name__ == "__main__":
    # Required for safe multiprocessing on Windows and occasionally Linux
    parent_dir = Path("/z/dat/person_reid/internal/input_videos")
    mp.freeze_support() 
    main(parent_dir, threshold=0.2, max_workers=32)