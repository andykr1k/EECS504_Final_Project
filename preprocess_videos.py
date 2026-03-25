"""
Takes a directory of directories with a config.yaml and a set of mp4s.

Splits the videos by cuts while maintaining each being less than max_scene_len_frames long.

Saves to a directory next to the videos with the same name as the original video but with "_processed" appended.
If this directory already exists, skip the video.
"""

from pathlib import Path
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg, FrameTimecode

from dataclasses import dataclass
import yaml

@dataclass
class Config:
    enabled: bool
    sam_prompt: str
    skip_frames: int
    max_scene_len_frames: int

def split_video(config: Config, input_video_path: Path, out_dir_path: Path):
    print(f"Processing {input_video_path}...")
    scene_list = detect(str(input_video_path), AdaptiveDetector(), show_progress=True)
    print(f"Found {len(scene_list)} natural scenes.")
    
    constrained_scenes = []
    
    # Enforce the max length constraint
    for start_time, end_time in scene_list:
        # Extract exact frame numbers
        fps = start_time.framerate
        start_frame = start_time.get_frames()
        end_frame = end_time.get_frames()
        
        # Subdivide the current scene into chunks of max_scene_len_frames
        current_start = start_frame
        while current_start < end_frame:
            # The chunk ends at either the max length limit, or the natural end of the scene
            current_end = min(current_start + config.max_scene_len_frames, end_frame)
            constrained_scenes.append((FrameTimecode(current_start, fps=fps), FrameTimecode(current_end, fps=fps)))
            current_start = current_end
            
    print(f"Adjusted to {len(constrained_scenes)} total scenes after length constraints.")

    # Create the output directory
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Define how the output files will be named
    output_template = str(out_dir_path / f"{input_video_path.stem}_$SCENE_NUMBER.mp4")
    
    # Run the actual splitting
    split_video_ffmpeg(
        input_video_path=str(input_video_path),
        scene_list=constrained_scenes,
        output_file_template=output_template,
        show_progress=True
    )

def split_dir(input_dir_path: Path):
    config_path = input_dir_path / "config.yaml"
    assert config_path.exists(), f"Config not found at {config_path}"
    config = Config(**yaml.safe_load(config_path.read_text()))

    if not config.enabled:
        print(f"Skipping {input_dir_path} (disabled)")
        return

    for video_path in input_dir_path.glob("*.mp4"):
        out_dir_path = video_path.parent / (video_path.stem + "_processed")
        if out_dir_path.exists():
            print(f"Skipping {video_path} (already processed)")
            continue
        split_video(config, video_path, out_dir_path)

if __name__ == "__main__":
    parent_dir = Path("/z/dat/person_reid/internal/input_videos")
    for dir_path in parent_dir.glob("*"):
        if dir_path.is_dir():
            split_dir(dir_path)
        
    
    
    