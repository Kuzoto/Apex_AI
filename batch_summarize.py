#!/usr/bin/env python3
"""
Batch Summarizer for Apex Legends Gameplay Data

This script processes all sample data collections and generates multiple types of summaries:
1. Raw video (no log) - using sum_test_efficient.py
2. Keyframes with unparsed log - using sum_test_efficient.py with --use_keyframes
3. Keyframes with parsed log - using sum_test_efficient.py with --use_keyframes --parse_json
4. Event frames with log - using sample_sum.py

Each summary is saved to its own text file in the collection's subfolder with
comprehensive metadata including token usage and processing details.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import argparse

def find_files_in_collection(collection_path):
    """
    Find video, log, and frames folder in a collection directory.
    
    Args:
        collection_path: Path to the collection directory
    
    Returns:
        dict with 'video', 'log', 'frames', and 'metadata' keys (None if not found)
    """
    collection_path = Path(collection_path)
    
    result = {
        'video': None,
        'log': None,
        'frames': None,
        'metadata': None
    }
    
    # Look for video file (common video extensions)
    # Prioritize gameplay_recording.mp4 first
    video_priority = ['gameplay_recording.mp4', 'gameplay_recording.avi', 'gameplay_recording.mov']
    for video_name in video_priority:
        video_path = collection_path / video_name
        if video_path.exists():
            result['video'] = video_path
            break
    
    # If not found, look for any video file
    if not result['video']:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        for ext in video_extensions:
            video_files = list(collection_path.glob(f'*{ext}'))
            if video_files:
                result['video'] = video_files[0]
                break
    
    # Look for log file - prioritize log_test.txt
    log_priority = ['log_test.txt', 'log.txt', 'events.log']
    for log_name in log_priority:
        log_path = collection_path / log_name
        if log_path.exists():
            result['log'] = log_path
            break
    
    # If not found, look for any log file (excluding metadata.txt)
    if not result['log']:
        log_files = list(collection_path.glob('*.txt')) + list(collection_path.glob('*.log'))
        log_files = [f for f in log_files if 'metadata' not in f.name.lower()]
        if log_files:
            # Prefer files with 'log' in the name
            log_with_log = [f for f in log_files if 'log' in f.name.lower()]
            result['log'] = log_with_log[0] if log_with_log else log_files[0]
    
    # Look for frames folder - prioritize 'frames' directory
    possible_frame_dirs = ['frames', 'event_frames', 'screenshots', 'images']
    for dir_name in possible_frame_dirs:
        frames_dir = collection_path / dir_name
        if frames_dir.exists() and frames_dir.is_dir():
            # Check if it actually contains image files
            image_files = (list(frames_dir.glob('*.png')) + 
                         list(frames_dir.glob('*.jpg')) + 
                         list(frames_dir.glob('*.jpeg')))
            if image_files:
                result['frames'] = frames_dir
                break
    
    # Look for metadata file
    metadata_path = collection_path / 'metadata.txt'
    if metadata_path.exists():
        result['metadata'] = metadata_path
    
    return result

def run_command_capture_output(cmd, description):
    """
    Run a command and capture both stdout and stderr.
    
    Args:
        cmd: Command list to run
        description: Description of what's being run
    
    Returns:
        Tuple of (returncode, stdout, stderr)
    """
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            encoding='utf-8',
            errors='replace'
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        
        return result.returncode, result.stdout, result.stderr
    
    except Exception as e:
        error_msg = f"Error running command: {e}"
        print(error_msg, file=sys.stderr)
        return -1, "", error_msg

def extract_usage_stats(output_text):
    """
    Extract token usage statistics from command output.
    
    Args:
        output_text: Combined stdout/stderr from command
    
    Returns:
        dict with usage stats or None
    """
    stats = {}
    
    # Look for usage statistics section
    lines = output_text.split('\n')
    in_stats_section = False
    
    for line in lines:
        if 'USAGE STATISTICS' in line:
            in_stats_section = True
            continue
        
        if in_stats_section:
            if '=' * 40 in line:  # End of section
                break
            
            # Parse lines like "Prompt tokens: 1,234"
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip().replace(',', '')
                
                try:
                    # Try to convert to int
                    stats[key] = int(value)
                except ValueError:
                    # Keep as string if not a number
                    stats[key] = value
    
    return stats if stats else None

def write_summary_file(output_path, metadata, summary_content, usage_stats=None, 
                       stdout="", stderr=""):
    """
    Write a summary file with metadata, usage stats, and content.
    
    Args:
        output_path: Path to write the summary file
        metadata: Dict with metadata (collection, type, timestamp, etc.)
        summary_content: The actual summary text
        usage_stats: Dict with token usage statistics (optional)
        stdout: Full stdout from the command (optional)
        stderr: Full stderr from the command (optional)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("APEX LEGENDS GAMEPLAY SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Write metadata
        f.write("METADATA:\n")
        f.write("-" * 80 + "\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Write usage statistics if available
        if usage_stats:
            f.write("TOKEN USAGE STATISTICS:\n")
            f.write("-" * 80 + "\n")
            for key, value in usage_stats.items():
                if isinstance(value, int):
                    f.write(f"{key}: {value:,}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
        
        # Write summary
        f.write("GAMEPLAY SUMMARY:\n")
        f.write("=" * 80 + "\n")
        f.write(summary_content)
        f.write("\n")
        f.write("=" * 80 + "\n\n")
        
        # Write full output logs (optional, for debugging)
        if stdout or stderr:
            f.write("\n\n")
            f.write("PROCESSING LOG:\n")
            f.write("=" * 80 + "\n")
            if stdout:
                f.write("STDOUT:\n")
                f.write(stdout)
                f.write("\n\n")
            if stderr:
                f.write("STDERR:\n")
                f.write(stderr)
                f.write("\n")

def process_collection(collection_path, scripts_path, skip_existing=False):
    """
    Process a single collection and generate all summary types.
    
    Args:
        collection_path: Path to the collection directory
        scripts_path: Path to directory containing the Python scripts
        skip_existing: If True, skip generating summaries that already exist
    
    Returns:
        dict with results for each summary type
    """
    collection_path = Path(collection_path)
    scripts_path = Path(scripts_path)
    
    collection_name = collection_path.name
    print(f"\n{'#' * 80}")
    print(f"# Processing Collection: {collection_name}")
    print(f"{'#' * 80}\n")
    
    # Find files
    files = find_files_in_collection(collection_path)
    
    print(f"Found files:")
    print(f"  Video: {files['video']}")
    print(f"  Log: {files['log']}")
    print(f"  Event Frames: {files['frames']}")
    
    if not files['video'] and not files['frames']:
        print(f"ERROR: No video or frames found in {collection_path}")
        return None
    
    results = {}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Summary 1: Raw video (with log)
    summary_type = "raw_video"
    output_file = collection_path / f"{summary_type}.txt"
    
    if skip_existing and output_file.exists():
        print(f"\nSkipping {summary_type} (already exists)")
        results[summary_type] = {"status": "skipped", "output": str(output_file)}
    elif files['video']:
        print(f"\n{'*' * 70}")
        print(f"* Generating Summary 1: Raw video (no log)")
        print(f"{'*' * 70}")
        
        cmd = [
            sys.executable,
            str(scripts_path / "sum_test_efficient.py"),
            "--video", str(files['video']),
            "--model", "gpt-5",
            "--output", str(output_file)
        ]
        
        returncode, stdout, stderr = run_command_capture_output(
            cmd, 
            "Raw video analysis (no log)"
        )
        
        if returncode == 0:
            usage_stats = extract_usage_stats(stdout + stderr)
            results[summary_type] = {
                "status": "success",
                "output": str(output_file),
                "usage": usage_stats
            }
            print(f"✓ Summary saved to: {output_file}")
        else:
            results[summary_type] = {
                "status": "failed",
                "error": stderr
            }
            print(f"✗ Failed to generate summary")
    else:
        results[summary_type] = {"status": "skipped", "reason": "no_video"}
    
    # Summary 2: Keyframes with unparsed log
    summary_type = "keyframes_log_unparsed"
    output_file = collection_path / f"{summary_type}.txt"
    
    if skip_existing and output_file.exists():
        print(f"\nSkipping {summary_type} (already exists)")
        results[summary_type] = {"status": "skipped", "output": str(output_file)}
    elif files['video'] and files['log']:
        print(f"\n{'*' * 70}")
        print(f"* Generating Summary 2: Keyframes with unparsed log")
        print(f"{'*' * 70}")
        
        cmd = [
            sys.executable,
            str(scripts_path / "sum_test_efficient.py"),
            "--video", str(files['video']),
            "--log", str(files['log']),
            "--use_keyframes",
            "--num_keyframes", "25",
            "--model", "gpt-5",
            "--output", str(output_file)
        ]
        
        returncode, stdout, stderr = run_command_capture_output(
            cmd,
            "Keyframes with unparsed log"
        )
        
        if returncode == 0:
            usage_stats = extract_usage_stats(stdout + stderr)
            results[summary_type] = {
                "status": "success",
                "output": str(output_file),
                "usage": usage_stats
            }
            print(f"✓ Summary saved to: {output_file}")
        else:
            results[summary_type] = {
                "status": "failed",
                "error": stderr
            }
            print(f"✗ Failed to generate summary")
    else:
        reason = "no_video" if not files['video'] else "no_log"
        results[summary_type] = {"status": "skipped", "reason": reason}
    
    # Summary 3: Keyframes with parsed log
    summary_type = "keyframes_log_parsed"
    output_file = collection_path / f"{summary_type}.txt"
    
    if skip_existing and output_file.exists():
        print(f"\nSkipping {summary_type} (already exists)")
        results[summary_type] = {"status": "skipped", "output": str(output_file)}
    elif files['video'] and files['log']:
        print(f"\n{'*' * 70}")
        print(f"* Generating Summary 3: Keyframes with parsed log")
        print(f"{'*' * 70}")
        
        cmd = [
            sys.executable,
            str(scripts_path / "sum_test_efficient.py"),
            "--video", str(files['video']),
            "--log", str(files['log']),
            "--parse_json",
            "--use_keyframes",
            "--num_keyframes", "25",
            "--model", "gpt-5",
            "--output", str(output_file)
        ]
        
        returncode, stdout, stderr = run_command_capture_output(
            cmd,
            "Keyframes with parsed log"
        )
        
        if returncode == 0:
            usage_stats = extract_usage_stats(stdout + stderr)
            results[summary_type] = {
                "status": "success",
                "output": str(output_file),
                "usage": usage_stats
            }
            print(f"✓ Summary saved to: {output_file}")
        else:
            results[summary_type] = {
                "status": "failed",
                "error": stderr
            }
            print(f"✗ Failed to generate summary")
    else:
        reason = "no_video" if not files['video'] else "no_log"
        results[summary_type] = {"status": "skipped", "reason": reason}
    
    # Summary 4: Event frames with log
    summary_type = "event_frames_log"
    output_file = collection_path / f"{summary_type}.txt"
    
    if skip_existing and output_file.exists():
        print(f"\nSkipping {summary_type} (already exists)")
        results[summary_type] = {"status": "skipped", "output": str(output_file)}
    elif files['frames'] and files['log']:
        print(f"\n{'*' * 70}")
        print(f"* Generating Summary 4: Event frames with log")
        print(f"{'*' * 70}")
        
        cmd = [
            sys.executable,
            str(scripts_path / "sample_sum.py"),
            "--log_file", str(files['log']),
            "--frames_folder", str(files['frames']),
            "--model", "gpt-5",
            "--output", str(output_file)
        ]
        
        returncode, stdout, stderr = run_command_capture_output(
            cmd,
            "Event frames with log"
        )
        
        if returncode == 0:
            usage_stats = extract_usage_stats(stdout + stderr)
            results[summary_type] = {
                "status": "success",
                "output": str(output_file),
                "usage": usage_stats
            }
            print(f"✓ Summary saved to: {output_file}")
        else:
            results[summary_type] = {
                "status": "failed",
                "error": stderr
            }
            print(f"✗ Failed to generate summary")
    else:
        reason = "no_frames" if not files['frames'] else "no_log"
        results[summary_type] = {"status": "skipped", "reason": reason}
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Generate gameplay summaries for a single collection"
    )
    
    parser.add_argument(
        "collection_path",
        help="Path to the collection directory to process"
    )
    
    parser.add_argument(
        "--scripts_path",
        default=".",
        help="Path to directory containing sample_sum.py and sum_test_efficient.py (default: current directory)"
    )
    
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generating summaries that already exist"
    )
    
    args = parser.parse_args()
    
    collection_path = Path(args.collection_path)
    scripts_path = Path(args.scripts_path)
    
    # Validate collection path
    if not collection_path.exists():
        print(f"ERROR: Collection directory not found: {collection_path}")
        sys.exit(1)
    
    if not collection_path.is_dir():
        print(f"ERROR: {collection_path} is not a directory")
        sys.exit(1)
    
    # Check for required scripts
    required_scripts = ["sample_sum.py", "sum_test_efficient.py"]
    for script in required_scripts:
        script_path = scripts_path / script
        if not script_path.exists():
            print(f"ERROR: Required script not found: {script_path}")
            sys.exit(1)
    
    print("=" * 80)
    print("APEX LEGENDS GAMEPLAY SUMMARIZER")
    print("=" * 80)
    print(f"Collection: {collection_path}")
    print(f"Scripts directory: {scripts_path}")
    print(f"Skip existing: {args.skip_existing}")
    print("=" * 80)
    print()
    
    # Process the collection
    try:
        results = process_collection(collection_path, scripts_path, args.skip_existing)
        
        # Print summary
        print("\n\n")
        print("=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"\nCollection: {collection_path.name}")
        
        if isinstance(results, dict) and "error" in results:
            print(f"  ✗ Collection failed: {results['error']}")
            sys.exit(1)
        else:
            print()
            for summary_type, result in results.items():
                status = result.get('status', 'unknown')
                if status == 'success':
                    usage = result.get('usage', {})
                    total_tokens = usage.get('Total tokens', 'N/A')
                    print(f"  ✓ {summary_type}: Success (tokens: {total_tokens})")
                elif status == 'skipped':
                    reason = result.get('reason', 'already exists')
                    print(f"  ○ {summary_type}: Skipped ({reason})")
                elif status == 'failed':
                    print(f"  ✗ {summary_type}: Failed")
        
        print("\n" + "=" * 80)
        print(f"All summaries have been saved to: {collection_path}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR processing collection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
