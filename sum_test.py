import cv2
import base64
import os
from openai import OpenAI
from pathlib import Path
import keyenv

def get_base64_size_mb(b64_string):
    """Calculate the size of a base64 string in MB."""
    # Base64 encoding increases size by ~33%, but we calculate actual size
    return len(b64_string) / (1024 * 1024)

def extract_frames_with_size_limit(video_path, max_size_mb=45, max_dimension=1280, max_frames=None):
    """
    Extract frames from a video while staying under a size limit.
    
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
    _, buffer = cv2.imencode('.jpg', sample_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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

def extract_frames_simple(video_path, num_frames=10):
    """
    Simple frame extraction with fixed number of frames.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
    
    Returns:
        List of base64-encoded frame images
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info: {total_frames} frames, {fps:.1f} fps, {duration:.2f}s duration")
    print(f"Extracting {num_frames} evenly-spaced frames...")
    
    # Calculate frame indices to extract evenly
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    frames_base64 = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Encode frame as JPEG with reasonable quality
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            frames_base64.append(frame_b64)
    
    cap.release()
    
    total_size_mb = sum(get_base64_size_mb(f) for f in frames_base64)
    print(f"Extracted {len(frames_base64)} frames, total size: {total_size_mb:.2f} MB")
    
    return frames_base64


def summarize_video_with_openai(frames_base64, api_key, model="gpt-5"):
    """
    Send frames to OpenAI API and get a video summary.
    
    Args:
        frames_base64: List of base64-encoded images
        api_key: OpenAI API key
        model: Model to use (gpt-4o, gpt-4o-mini, etc.)
    
    Returns:
        Summary text from OpenAI
    """
    client = OpenAI(api_key=api_key)
    
    # Prepare the messages with images
    content = [
        {
            "type": "text",
            "text": "These images are frames extracted from a video. Please analyze them and provide a comprehensive summary of what happens in the video, including key events, actions, people, objects, and any other relevant details."
        }
    ]
    
    # Add each frame as an image
    # for frame_b64 in frames_base64:
    #     content.append({
    #         "type": "image_url",
    #         "image_url": {
    #             "url": f"data:image/jpeg;base64,{frame_b64}"
    #         },
    #         "detail": "auto"
    #     })
    
    for frame_b64 in frames_base64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_b64}",
                "detail": "low"  # Use "low" for faster/cheaper processing, "high" for more detail
            }
        })
    
    print(f"Sending {len(frames_base64)} frames to OpenAI API...")
    
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

    print(response)
    
    summary = response.choices[0].message.content
    return summary

def main():
    # Configuration
    VIDEO_PATH = "sample_data\collection_20251119_165050\gameplay_recording.mp4"  # Change this to your video path
    API_KEY = os.getenv("OPENAI_API_KEY")  # Set this in your environment
    USE_SMART_EXTRACTION = True  # Set to True to auto-calculate frames based on size
    MAX_SIZE_MB = 45  # Maximum total size in MB (stay under 50MB limit)
    MAX_DIMENSION = 1280  # Resize frames to this max width/height (smaller = smaller files)
    MAX_FRAMES = 500  # Maximum frames if using smart extraction
    SIMPLE_NUM_FRAMES = 50  # Number of frames if using simple extraction
    
    # Resolution presets:
    # 1920 (Full HD) - ~0.4-0.6 MB per frame - good for ~75-110 frames
    # 1280 (HD) - ~0.2-0.3 MB per frame - good for ~150-225 frames
    # 960 - ~0.1-0.15 MB per frame - good for ~300-450 frames
    # 640 (SD) - ~0.05-0.08 MB per frame - good for ~560-900 frames
    
    if not API_KEY:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    if not Path(VIDEO_PATH).exists():
        raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")
    
    # Extract frames from video
    print(f"Processing video: {VIDEO_PATH}\n")
    
    if USE_SMART_EXTRACTION:
        frames = extract_frames_with_size_limit(
            VIDEO_PATH, 
            max_size_mb=MAX_SIZE_MB,
            max_dimension=MAX_DIMENSION,
            max_frames=MAX_FRAMES
        )
    else:
        frames = extract_frames_simple(VIDEO_PATH, num_frames=SIMPLE_NUM_FRAMES)
    
    if not frames:
        print("Error: No frames extracted!")
        return
    
    # Get summary from OpenAI
    print("\n" + "="*60)
    summary = summarize_video_with_openai(frames, API_KEY, "gpt-5")
    
    # Print results
    print("="*60)
    print("VIDEO SUMMARY")
    print("="*60)
    print(summary)
    print("="*60)

if __name__ == "__main__":
    main()