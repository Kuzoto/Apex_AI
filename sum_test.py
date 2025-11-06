import cv2
import base64
import os
from openai import OpenAI
from pathlib import Path
import keyenv

def extract_frames(video_path, num_frames=10, extract_all=False):
    """
    Extract frames from a video.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract (if extract_all is False)
        extract_all: If True, extract every single frame
    
    Returns:
        List of base64-encoded frame images
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    frames_base64 = []
    
    if extract_all:
        # Extract every frame
        print(f"Extracting all {total_frames} frames...")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            # Convert to base64
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            frames_base64.append(frame_b64)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...")
    else:
        # Calculate frame indices to extract evenly
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                # Convert to base64
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                frames_base64.append(frame_b64)
    
    cap.release()
    
    print(f"Extracted {len(frames_base64)} frames from video")
    print(f"Video duration: {duration:.2f} seconds")
    
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
        max_completion_tokens=4000
    )

    print(response)
    
    summary = response.choices[0].message.content
    return summary

def main():
    # Configuration
    VIDEO_PATH = "Untitled_video_1.mp4"  # Change this to your video path
    API_KEY = os.getenv("OPENAI_API_KEY")  # Set this in your environment
    NUM_FRAMES = 500  # Number of frames to extract (ignored if EXTRACT_ALL is True)
    EXTRACT_ALL = False  # Set to True to extract every frame
    
    if not API_KEY:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    if not Path(VIDEO_PATH).exists():
        raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")
    
    # Extract frames from video
    print(f"Processing video: {VIDEO_PATH}")
    frames = extract_frames(VIDEO_PATH, num_frames=NUM_FRAMES, extract_all=EXTRACT_ALL)
    
    # Get summary from OpenAI
    summary = summarize_video_with_openai(frames, API_KEY)
    
    # Print results
    print("\n" + "="*50)
    print("VIDEO SUMMARY")
    print("="*50)
    print(summary)
    print("="*50)

if __name__ == "__main__":
    main()