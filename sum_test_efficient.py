"""
Apex Legends Gameplay Video Analyzer (Efficient Version)

This script analyzes gameplay videos with optional log file support.
It supports multiple optimization features:

Frame Extraction:
1. Uniform sampling: Extracts evenly-spaced frames from the video
2. Keyframe extraction: Uses video-keyframe-detector to intelligently extract the most 
   representative frames based on peak detection of frame differences
   (reduces redundancy and focuses on important moments - can reduce frames by 60-80%)

Log Processing:
1. Plain text: Reads log file as-is (default)
2. JSON parsing: Parses and formats JSON game events with --parse_json flag

Usage:
    # Basic usage (plain text log, uniform sampling)
    python sum_test_efficient.py --video gameplay.mp4 --log log.txt

    # With JSON parsing
    python sum_test_efficient.py --video gameplay.mp4 --log log.txt --parse_json

    # With keyframe extraction (install: pip install video-keyframe-detector matplotlib)
    python sum_test_efficient.py --video gameplay.mp4 --log log.txt --use_keyframes --num_keyframes 25
    
    # All optimizations enabled
    python sum_test_efficient.py --video gameplay.mp4 --log log.txt --parse_json --use_keyframes --num_keyframes 25
"""

import cv2
import base64
import os
import json
import argparse
import tempfile
import shutil
from pathlib import Path
from openai import OpenAI
import keyenv

# Try to import video-keyframe-detector for keyframe extraction
try:
    from KeyFrameDetector.key_frame_detector import keyframeDetection
    KEYFRAME_DETECTOR_AVAILABLE = True
except ImportError:
    KEYFRAME_DETECTOR_AVAILABLE = False
    print("Warning: video-keyframe-detector not installed. Install with 'pip install video-keyframe-detector' for keyframe extraction.")

def get_base64_size_mb(b64_string):
    """Calculate the size of a base64 string in MB."""
    return len(b64_string) / (1024 * 1024)

def parse_json_log(log_path):
    """
    Parse JSON data from a log file where each line is a JSON object.
    
    Args:
        log_path: Path to the log file containing JSON data
    
    Returns:
        Tuple of (raw_text, parsed_events_list)
    """
    log_path = Path(log_path)
    
    if not log_path.exists():
        print(f"Warning: Log file not found: {log_path}")
        return "", []
    
    parsed_events = []
    raw_lines = []
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                raw_lines.append(line)
                
                try:
                    # Parse JSON from each line
                    event = json.loads(line)
                    parsed_events.append(event)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON on line {line_num}: {e}")
                    print(f"  Line content: {line[:100]}...")
        
        raw_text = '\n'.join(raw_lines)
        print(f"Parsed {len(parsed_events)} JSON events from {log_path}")
        return raw_text, parsed_events
    
    except Exception as e:
        print(f"Error reading log file: {e}")
        return "", []

def format_game_events(parsed_events):
    """
    Format parsed JSON events into a human-readable summary.
    
    Args:
        parsed_events: List of parsed JSON objects
    
    Returns:
        Formatted string describing the events
    """
    if not parsed_events:
        return "No game events recorded."
    
    summary_parts = []
    summary_parts.append(f"Total Events: {len(parsed_events)}\n")
    
    # Categorize events
    game_info = []
    player_info = []
    weapons = []
    inventory = []
    other = []
    
    for event in parsed_events:
        if 'game_info' in event:
            game_info.append(event['game_info'])
        elif 'me' in event:
            player_data = event['me']
            
            # Check what type of player data
            if 'weapons' in player_data:
                try:
                    weapons_data = json.loads(player_data['weapons'])
                    weapons.append(weapons_data)
                except:
                    weapons.append(player_data['weapons'])
            elif 'inUse' in player_data:
                try:
                    inuse_data = json.loads(player_data['inUse'])
                    player_info.append(f"Using: {inuse_data.get('inUse', 'unknown')}")
                except:
                    player_info.append(f"Using: {player_data['inUse']}")
            elif 'ultimate_cooldown' in player_data:
                player_info.append(player_data)
            elif any(k.startswith('inventory_') for k in player_data.keys()):
                inventory.append(player_data)
            else:
                player_info.append(player_data)
        else:
            other.append(event)
    
    # Format game info
    if game_info:
        summary_parts.append("=== GAME STATE ===")
        for info in game_info:
            if isinstance(info, dict):
                for k, v in info.items():
                    summary_parts.append(f"  {k}: {v}")
            else:
                summary_parts.append(f"  {info}")
        summary_parts.append("")
    
    # Format weapons
    if weapons:
        summary_parts.append("=== WEAPONS ===")
        for weapon_set in weapons:
            if isinstance(weapon_set, dict):
                for slot, weapon in weapon_set.items():
                    summary_parts.append(f"  {slot}: {weapon}")
            else:
                summary_parts.append(f"  {weapon_set}")
        summary_parts.append("")
    
    # Format player info
    if player_info:
        summary_parts.append("=== PLAYER STATUS ===")
        for info in player_info:
            if isinstance(info, str):
                summary_parts.append(f"  {info}")
            elif isinstance(info, dict):
                for k, v in info.items():
                    summary_parts.append(f"  {k}: {v}")
        summary_parts.append("")
    
    # Format inventory
    if inventory:
        summary_parts.append("=== INVENTORY ===")
        for inv_data in inventory:
            for slot, item_json in inv_data.items():
                try:
                    item = json.loads(item_json)
                    name = item.get('name', 'unknown')
                    amount = item.get('amount', '?')
                    summary_parts.append(f"  {slot}: {name} x{amount}")
                except:
                    summary_parts.append(f"  {slot}: {item_json}")
        summary_parts.append("")
    
    # Format other events
    if other:
        summary_parts.append("=== OTHER EVENTS ===")
        for event in other:
            summary_parts.append(f"  {event}")
        summary_parts.append("")
    
    return '\n'.join(summary_parts)

def extract_keyframes_with_detector(video_path, num_keyframes=50, max_dimension=1280, threshold=0.6):
    """
    Extract keyframes from video using video-keyframe-detector library.
    
    This detector uses peak detection on frame differences to identify the most
    representative and significant frames that describe movement or main events.
    
    Args:
        video_path: Path to the video file
        num_keyframes: Maximum number of keyframes to extract
        max_dimension: Maximum width or height for resizing
        threshold: Threshold for peak detection (0.0-1.0, higher = fewer keyframes)
    
    Returns:
        List of base64-encoded keyframe images
    """
    if not KEYFRAME_DETECTOR_AVAILABLE:
        print("video-keyframe-detector not available, falling back to uniform frame extraction")
        return extract_frames_with_size_limit(video_path, max_dimension=max_dimension, max_frames=num_keyframes)
    
    print(f"Extracting up to {num_keyframes} keyframes using video-keyframe-detector...")
    
    # Create temporary directory for keyframes
    temp_dir = tempfile.mkdtemp(prefix="keyframes_")
    
    try:
        # Get video info first
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        print(f"Video info: {total_frames} frames, {fps:.1f} fps, {duration:.2f}s duration")
        
        # Extract keyframes using the detector
        print(f"Detecting keyframes (threshold={threshold})...")
        keyframeDetection(
            source=str(video_path),
            dest=temp_dir,
            Thres=threshold,
            plotMetrics=False,
            verbose=False
        )
        
        # Load extracted keyframes (they're saved in a 'keyFrames' subdirectory)
        keyframes_dir = Path(temp_dir) / "keyFrames"
        keyframe_files = sorted(keyframes_dir.glob("*.jpg")) if keyframes_dir.exists() else []
        
        if not keyframe_files:
            print(f"Warning: No keyframes extracted, trying lower threshold...")
            # Try with lower threshold
            keyframeDetection(
                source=str(video_path),
                dest=temp_dir,
                Thres=max(0.2, threshold - 0.3),
                plotMetrics=False,
                verbose=False
            )
            keyframe_files = sorted(keyframes_dir.glob("*.jpg")) if keyframes_dir.exists() else []
        
        if not keyframe_files:
            print("Warning: Still no keyframes extracted, falling back to uniform sampling")
            return extract_frames_with_size_limit(
                video_path, 
                max_dimension=max_dimension, 
                max_frames=num_keyframes
            )
        
        print(f"Extracted {len(keyframe_files)} keyframes")
        
        # Limit to requested number of keyframes if we got more
        if len(keyframe_files) > num_keyframes:
            print(f"Limiting from {len(keyframe_files)} to {num_keyframes} keyframes")
            # Select evenly spaced keyframes
            indices = [int(i * len(keyframe_files) / num_keyframes) for i in range(num_keyframes)]
            keyframe_files = [keyframe_files[i] for i in indices]
        
        print(f"Processing {len(keyframe_files)} keyframes")
        
        frames_base64 = []
        total_size_mb = 0
        
        for i, img_path in enumerate(keyframe_files):
            # Read image
            img = cv2.imread(str(img_path))
            
            if img is None:
                continue
            
            # Get original dimensions
            orig_height, orig_width = img.shape[0], img.shape[1]
            
            # Resize if needed
            if max(orig_height, orig_width) > max_dimension:
                scale = max_dimension / max(orig_height, orig_width)
                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Print resize info for first frame
                if i == 0:
                    print(f"Resizing frames: {orig_width}x{orig_height} → {new_width}x{new_height}")
            
            # Encode as JPEG with good quality/size balance
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            frame_size_mb = get_base64_size_mb(frame_b64)
            
            frames_base64.append(frame_b64)
            total_size_mb += frame_size_mb
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(keyframe_files)} keyframes ({total_size_mb:.1f} MB so far)")
        
        print(f"\nLoaded {len(frames_base64)} keyframes")
        print(f"Total size: {total_size_mb:.2f} MB")
        print(f"Average frame size: {total_size_mb/len(frames_base64):.3f} MB")
        
        return frames_base64
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not remove temp directory {temp_dir}: {e}")

def extract_frames_with_size_limit(video_path, max_size_mb=45, max_dimension=1280, max_frames=None):
    """
    Extract frames from a video while staying under a size limit (uniform sampling).
    
    Args:
        video_path: Path to the video file
        max_size_mb: Maximum total size in MB (default 45MB to stay safely under 50MB)
        max_dimension: Maximum width or height (frames will be scaled down if larger)
        max_frames: Maximum number of frames to extract (None = auto-calculate)
    
    Returns:
        List of base64-encoded frame images
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {total_frames} frames, {fps:.1f} fps, {duration:.2f}s duration")
    print(f"Original resolution: {width}x{height}")
    
    # First, extract a sample frame to estimate size
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, sample_frame = cap.read()
    
    if not ret:
        print("Error: Could not read sample frame")
        cap.release()
        return []
    
    # Resize sample frame if needed
    if max(sample_frame.shape[0], sample_frame.shape[1]) > max_dimension:
        scale = max_dimension / max(sample_frame.shape[0], sample_frame.shape[1])
        new_width = int(sample_frame.shape[1] * scale)
        new_height = int(sample_frame.shape[0] * scale)
        sample_frame = cv2.resize(sample_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Frames will be resized to: {new_width}x{new_height}")
    
    # Encode sample frame and check size
    _, buffer = cv2.imencode('.jpg', sample_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    sample_b64 = base64.b64encode(buffer).decode('utf-8')
    sample_size_mb = get_base64_size_mb(sample_b64)
    
    print(f"Sample frame size: {sample_size_mb:.3f} MB (max dimension: {max_dimension})")
    
    # Calculate how many frames we can fit
    estimated_frames = int(max_size_mb / sample_size_mb)
    
    if max_frames is not None:
        estimated_frames = min(estimated_frames, max_frames)
    
    # Ensure we don't exceed total frames
    num_frames = min(estimated_frames, total_frames)
    
    print(f"Extracting {num_frames} frames (estimated {num_frames * sample_size_mb:.1f} MB total)")
    
    # Calculate frame indices to extract evenly
    if num_frames >= total_frames:
        # Extract all frames
        frame_indices = list(range(total_frames))
    else:
        # Extract evenly spaced frames
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    frames_base64 = []
    total_size_mb = 0
    
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Resize frame if needed
        if max(frame.shape[0], frame.shape[1]) > max_dimension:
            scale = max_dimension / max(frame.shape[0], frame.shape[1])
            new_width = int(frame.shape[1] * scale)
            new_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        frame_size_mb = get_base64_size_mb(frame_b64)
        
        # Check if adding this frame would exceed the limit
        if total_size_mb + frame_size_mb > max_size_mb:
            print(f"Reached size limit at frame {i+1}/{num_frames}")
            break
        
        frames_base64.append(frame_b64)
        total_size_mb += frame_size_mb
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{num_frames} frames ({total_size_mb:.1f} MB so far)")
    
    cap.release()
    
    print(f"\nExtracted {len(frames_base64)} frames")
    print(f"Total size: {total_size_mb:.2f} MB")
    print(f"Average frame size: {total_size_mb/len(frames_base64):.3f} MB")
    
    return frames_base64

def summarize_video_with_openai(frames_base64, game_events_text, api_key, model="gpt-5"):
    """
    Send frames and game events to OpenAI API and get a comprehensive summary.
    
    Args:
        frames_base64: List of base64-encoded images
        game_events_text: Formatted text describing game events from JSON log
        api_key: OpenAI API key
        model: Model to use (gpt-5, gpt-4o, etc.)
    
    Returns:
        Summary text from OpenAI
    """
    client = OpenAI(api_key=api_key)
    
    # Prepare the messages with game context and images
    content = []
    
    if game_events_text:
        content.append({
            "type": "text",
            "text": f"""You are analyzing gameplay from Apex Legends. Below is structured game data extracted from the game's API during this video clip:

{game_events_text}

Now, please analyze the video frames that follow. Provide a detailed summary that:
1. Describes the key actions and events shown in the video
2. Correlates the visual gameplay with the game data above (weapons used, game phase, inventory, etc.)
3. Identifies important moments like combat, looting, movement, team interactions
4. Notes the sequence of events and overall outcome

Be specific and detailed in your analysis."""
        })
    else:
        content.append({
            "type": "text",
            "text": "These images are frames extracted from an Apex Legends gameplay video. Please analyze them and provide a comprehensive summary of what happens, including key events, actions, combat, and any other relevant gameplay details."
        })
    
    # Add each frame as an image
    for frame_b64 in frames_base64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_b64}",
                "detail": "low"  # Use "low" for faster/cheaper processing, "high" for more detail
            }
        })
    
    print(f"\nSending {len(frames_base64)} frames to OpenAI API...")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        max_completion_tokens=8000
    )

    # Print usage information
    if response.usage:
        print("\n" + "="*60)
        print("USAGE STATISTICS")
        print("="*60)
        print(f"Prompt tokens: {response.usage.prompt_tokens:,}")
        print(f"Completion tokens: {response.usage.completion_tokens:,}")
        print(f"Total tokens: {response.usage.total_tokens:,}")
        print("="*60 + "\n")
    
    summary = response.choices[0].message.content
    return summary

def main():
    parser = argparse.ArgumentParser(
        description="Summarize Apex Legends gameplay video with JSON log data using OpenAI"
    )
    
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the gameplay video file"
    )
    
    parser.add_argument(
        "--log",
        help="Path to the JSON log file (optional)"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="OpenAI model to use (default: gpt-5)"
    )
    
    parser.add_argument(
        "--max_size_mb",
        type=float,
        default=45,
        help="Maximum total size in MB for frames (default: 45)"
    )
    
    parser.add_argument(
        "--max_dimension",
        type=int,
        default=1280,
        help="Maximum frame dimension (default: 1280)"
    )
    
    parser.add_argument(
        "--max_frames",
        type=int,
        help="Maximum number of frames to extract (optional)"
    )
    
    parser.add_argument(
        "--use_keyframes",
        action="store_true",
        help="Use video-keyframe-detector for intelligent keyframe extraction (requires 'pip install video-keyframe-detector')"
    )
    
    parser.add_argument(
        "--num_keyframes",
        type=int,
        default=50,
        help="Maximum number of keyframes to extract (default: 50)"
    )
    
    parser.add_argument(
        "--keyframe_threshold",
        type=float,
        default=0.6,
        help="Threshold for keyframe detection (0.0-1.0, higher = fewer keyframes, default: 0.6)"
    )
    
    parser.add_argument(
        "--parse_json",
        action="store_true",
        help="Parse JSON log data and format it (default: treat log as plain text)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file path for the summary (optional)"
    )
    
    args = parser.parse_args()
    
    # Get API key from environment
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    if not Path(args.video).exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    print("="*60)
    print("APEX LEGENDS GAMEPLAY ANALYZER")
    print("="*60)
    print(f"Video: {args.video}")
    print(f"Log: {args.log if args.log else 'None'}")
    print(f"JSON Parsing: {'Enabled' if args.parse_json else 'Disabled (plain text)'}")
    print(f"Model: {args.model}")
    print("="*60)
    print()
    
    # Process log file if provided
    game_events_text = ""
    if args.log:
        if args.parse_json:
            # Parse as JSON and format
            raw_text, parsed_events = parse_json_log(args.log)
            game_events_text = format_game_events(parsed_events)
            
            if game_events_text:
                print("\n" + "="*60)
                print("PARSED GAME EVENTS (JSON)")
                print("="*60)
                print(game_events_text)
                print("="*60)
                print()
        else:
            # Read as plain text
            try:
                with open(args.log, 'r', encoding='utf-8') as f:
                    game_events_text = f.read()
                
                if game_events_text:
                    print("\n" + "="*60)
                    print("LOG CONTENT (Plain Text)")
                    print("="*60)
                    print(game_events_text[:500])  # Show first 500 chars
                    if len(game_events_text) > 500:
                        print(f"... ({len(game_events_text)} total characters)")
                    print("="*60)
                    print()
            except Exception as e:
                print(f"Warning: Could not read log file: {e}")
                game_events_text = ""
    
    # Extract frames from video
    print("Processing video...\n")
    
    if args.use_keyframes:
        # Use video-keyframe-detector for intelligent keyframe extraction
        frames = extract_keyframes_with_detector(
            args.video,
            num_keyframes=args.num_keyframes,
            max_dimension=args.max_dimension,
            threshold=args.keyframe_threshold
        )
    else:
        # Use uniform frame extraction
        frames = extract_frames_with_size_limit(
            args.video,
            max_size_mb=args.max_size_mb,
            max_dimension=args.max_dimension,
            max_frames=args.max_frames
        )
    
    if not frames:
        print("Error: No frames extracted!")
        return
    
    # Get summary from OpenAI
    print("\n" + "="*60)
    summary = summarize_video_with_openai(frames, game_events_text, API_KEY, args.model)
    
    # Print results
    print("="*60)
    print("GAMEPLAY SUMMARY")
    print("="*60)
    print(summary)
    print("="*60)
    
    # Save to file if output path specified
    if args.output:
        output_path = Path(args.output)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Video: {args.video}\n")
                f.write(f"Log: {args.log if args.log else 'None'}\n")
                f.write(f"Number of Frames: {len(frames)}\n")
                f.write(f"Model: {args.model}\n")
                f.write("\n" + "="*60 + "\n\n")
                
                if game_events_text:
                    f.write("GAME EVENTS DATA:\n")
                    f.write("="*60 + "\n")
                    f.write(game_events_text)
                    f.write("\n" + "="*60 + "\n\n")
                
                f.write("GAMEPLAY SUMMARY:\n")
                f.write("="*60 + "\n")
                f.write(summary)
            
            print(f"\nSummary saved to: {output_path}")
        except Exception as e:
            print(f"\nError saving summary to file: {e}")

if __name__ == "__main__":
    main()

