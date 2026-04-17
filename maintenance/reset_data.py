"""
Crawls the input video directory and removes segmentations and preprocessed slices in parallel.
"""

from pathlib import Path
import subprocess
import shutil
import multiprocessing
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed

def remove_directory(directory: Path | str, use_parallel: bool = True, dry_run: bool = False):
    directory_str = str(directory)
    print(f"Queueing deletion for: {directory_str}")
    
    if use_parallel:
        # Get core count for xargs. Maxing this out is usually best for I/O bound unlinks.
        cores = multiprocessing.cpu_count()
        safe_path = shlex.quote(directory_str)
        
        # Step 1: Nuke files in parallel using xargs
        cmd_files = f"find {safe_path} -type f -print0 | xargs -0 -P {cores} -n 10000 rm -f"
        # Step 2: Remove the now-empty directory structure
        cmd_dirs = f"rm -rf {safe_path}"
        
        if not dry_run:
            # shell=True is required for the pipe (|), shlex.quote keeps it secure.
            subprocess.run(cmd_files, shell=True, check=True)
            subprocess.run(cmd_dirs, shell=True, check=True)
            return f"Successfully removed {directory_str}"
        else:
            return f"[DRY RUN] Would execute: {cmd_files} && {cmd_dirs}"
    else:
        if not dry_run:
            shutil.rmtree(directory)
            return f"Successfully removed {directory_str} via shutil"
        else:
            return f"[DRY RUN] Would execute shutil.rmtree({directory_str})"

def main(input_dir: Path | str, remove_segmentations: bool, remove_preprocessed_slices: bool, use_parallel: bool = True, dry_run: bool = False):
    input_path = Path(input_dir)
    directories_to_remove = []

    # Gather all target directories first
    if remove_preprocessed_slices:
        directories_to_remove.extend(list(input_path.rglob("*_processed")))

    if remove_segmentations:
        directories_to_remove.extend(list(input_path.rglob("*_segmentations")))

    # Deduplicate in case there's any weird overlap
    directories_to_remove = list(set(directories_to_remove))
    
    if not directories_to_remove:
        print("No matching directories found.")
        return

    print(f"Found {len(directories_to_remove)} directories to process.\n")

    # Process multiple directories simultaneously using a thread pool
    # Max workers is capped at 4 here to avoid overwhelming the disk controller, 
    # but you can tune this up or down depending on your specific NVMe/HDD setup.
    max_concurrent_dirs = 4 
    
    with ThreadPoolExecutor(max_workers=max_concurrent_dirs) as executor:
        # Submit all tasks to the executor
        futures = {
            executor.submit(remove_directory, dir_path, use_parallel, dry_run): dir_path 
            for dir_path in directories_to_remove
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                dir_path = futures[future]
                print(f"Directory {dir_path} generated an exception: {exc}")

if __name__ == "__main__":
    remove_segmentations = True
    remove_preprocessed_slices = False

    use_parallel = True
    dry_run = False
    
    # main("/z/dat/person_reid/train/internal/input_videos", remove_segmentations, remove_preprocessed_slices, use_parallel, dry_run)
    # main("/z/dat/person_reid/val/internal/input_videos", remove_segmentations, remove_preprocessed_slices, use_parallel, dry_run)
    main("/z/dat/person_reid/train/person_reid_in_the_wild", remove_segmentations, remove_preprocessed_slices, use_parallel, dry_run)
