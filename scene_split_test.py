from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg, ContentDetector, ThresholdDetector

from pathlib import Path

def main(videos_path: Path, splits_path: Path):
    all_videos = []
    for video_path in videos_path.glob("*.webm"):
        all_videos.append(video_path)
    for video_path in videos_path.glob("*.mp4"):
        all_videos.append(video_path)

    for video_path in all_videos:
        out_path = splits_path / video_path.stem
        if out_path.exists():
            print(f"Skipping {video_path} as it has already been processed")
            continue

        print(f"Processing {video_path}")
        scene_list = detect(str(video_path), AdaptiveDetector(), show_progress=True)
        # scene_list = detect(str(video_path), ContentDetector())
        # scene_list = detect(str(video_path), ThresholdDetector(threshold=100.0), show_progress=True)
        print(f"Found {len(scene_list)} scenes")
        for i, scene in enumerate(scene_list):
            print(f"Scene {i}: {scene}")
        out_path.mkdir(parents=True, exist_ok=True)
        split_video_ffmpeg(str(video_path), scene_list, output_dir=str(out_path))

if __name__ == "__main__":
    data_path = Path("data")
    videos_path = data_path / "test_videos"
    splits_path = data_path / "test_splits"
    main(videos_path, splits_path)
    