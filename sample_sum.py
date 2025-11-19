import cv2
import base64
import os
import sys
import argparse
from pathlib import Path
from openai import OpenAI
import keyenv

def read_log_file(log_path):
    """
    Read the contents of a log file.
    
    Args:
        log_path: Path to the log file
    
    Returns:
        String content of the log file
    """
    log_path = Path(log_path)
    
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return ""
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Read log file: {log_path} ({len(content)} characters)")
        return content
    
    except Exception as e:
        print(f"Error reading log file: {e}")
        return ""

def load_frames_from_folder(frames_folder):
    """
    Load all image frames from a folder as base64.
    
    Args:
        frames_folder: Path to folder containing image frames
    
    Returns:
        List of dicts with 'filename' and 'data' keys
    """
    frames_folder = Path(frames_folder)
    
    if not frames_folder.exists():
        print(f"Error: Frames folder not found: {frames_folder}")
        return []
    
    if not frames_folder.is_dir():
        print(f"Error: {frames_folder} is not a directory")
        return []
    
    frames_base64 = []
    
    # Support multiple image formats
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(frames_folder.glob(ext))
    
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} image file(s) in {frames_folder}")
    
    for img_path in image_files:
        try:
            print(f"  Loading: {img_path.name}")
            
            # Read image with OpenCV
            img = cv2.imread(str(img_path))
            
            if img is not None:
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', img)
                # Convert to base64
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                frames_base64.append({
                    'filename': img_path.name,
                    'data': frame_b64
                })
            else:
                print(f"  Warning: Could not read image: {img_path.name}")
        
        except Exception as e:
            print(f"  Error loading {img_path.name}: {e}")
    
    print(f"Successfully loaded {len(frames_base64)} frame(s)")
    return frames_base64

def summarize_with_openai(log_content, frames_base64, api_key, model="gpt-5"):
    """
    Send log content and frames to OpenAI API for summarization.
    
    Args:
        log_content: Text content from log file
        frames_base64: List of dicts with 'filename' and 'data' keys
        api_key: OpenAI API key
        model: OpenAI model to use
    
    Returns:
        Summary text from OpenAI
    """
    if not log_content and not frames_base64:
        return "No content to summarize (both log and frames are empty)."
    
    client = OpenAI(api_key=api_key)
    
    # Build the content array
    content = []
    
    # Add log context
    if log_content:
        content.append({
            "type": "text",
            "text": f"""You are analyzing gameplay events from Apex Legends. Below is the log data showing game events:

{log_content}

Please analyze the screenshots that follow and provide a detailed summary of what happened during this gameplay session. Focus on key events like kills, knockdowns, damage dealt, team status changes, and match progression. Be specific about the sequence of events."""
        })
    else:
        content.append({
            "type": "text",
            "text": "Please analyze the following gameplay screenshots and provide a summary of the events shown."
        })
    
    # Add frames
    print(f"\nPreparing to send {len(frames_base64)} frame(s) to OpenAI...")
    for i, frame_info in enumerate(frames_base64, 1):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_info['data']}",
                "detail": "low"  # Use "low" for speed, "high" for detail
            }
        })
        if i % 10 == 0:
            print(f"  Added {i}/{len(frames_base64)} frames...")
    
    print(f"Sending request to OpenAI ({model})...")
    
    try:
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
        
        print("Received response from OpenAI")
        summary = response.choices[0].message.content
        return summary
    
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Summarize Apex Legends gameplay from log file and screenshots using OpenAI"
    )
    
    parser.add_argument(
        "--log_file",
        help="Path to the log file containing game events"
    )
    
    parser.add_argument(
        "--frames_folder",
        help="Path to the folder containing screenshot frames"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="OpenAI model to use (default: gpt-5)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file path for the summary (optional)"
    )
    
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    print("="*60)
    print("APEX LEGENDS GAMEPLAY SUMMARIZER")
    print("="*60)
    print(f"Log file: {args.log_file}")
    print(f"Frames folder: {args.frames_folder}")
    print(f"Model: {args.model}")
    print("="*60)
    print()
    
    # Read log content
    log_content = read_log_file(args.log_file)
    
    # Load frames
    frames = load_frames_from_folder(args.frames_folder)
    
    # Check if we have content to summarize
    if not log_content and not frames:
        print("\nError: No content found to summarize")
        sys.exit(1)
    
    # Generate summary
    print("\n" + "="*60)
    summary = summarize_with_openai(log_content, frames, api_key, args.model)
    
    if summary:
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
                    f.write(f"Log File: {args.log_file}\n")
                    f.write(f"Frames Folder: {args.frames_folder}\n")
                    f.write(f"Number of Frames: {len(frames)}\n")
                    f.write(f"Log Size: {len(log_content)} characters\n")
                    f.write(f"Model: {args.model}\n")
                    f.write("\n" + "="*60 + "\n\n")
                    f.write(summary)
                print(f"\nSummary saved to: {output_path}")
            except Exception as e:
                print(f"\nError saving summary to file: {e}")
    else:
        print("Failed to generate summary")
        sys.exit(1)

if __name__ == "__main__":
    main()