import cv2
import base64
import os
import time
import subprocess
import signal
import sys
from pathlib import Path
from openai import OpenAI
from datetime import datetime
import keyenv

class EventSummarizer:
    def __init__(self, log_file="log_test.txt", frames_folder="event_frames", 
                 api_key=None, interval=10):
        """
        Initialize the event summarizer.
        
        Args:
            log_file: Path to the log text file
            frames_folder: Path to the event frames folder
            api_key: OpenAI API key
            interval: Time in seconds between summaries
        """
        self.log_file = Path(log_file)
        self.frames_folder = Path(frames_folder)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.interval = interval
        self.event_cap_process = None
        self.running = False
        
        if not self.api_key:
            raise ValueError("Please provide OpenAI API key or set OPENAI_API_KEY environment variable")
    
    def start_event_capture(self):
        """Start the event_cap.py script as a subprocess."""
        print("Starting event capture script...")
        self.event_cap_process = subprocess.Popen(
            [sys.executable, "event_cap.py", str(self.log_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Event capture script started.")
    
    def stop_event_capture(self):
        """Stop the event_cap.py script."""
        if self.event_cap_process:
            print("\nStopping event capture script...")
            self.event_cap_process.send_signal(signal.SIGINT)
            self.event_cap_process.wait(timeout=5)
            print("Event capture script stopped.")
    
    def read_log_content(self):
        """Read the contents of the log file."""
        if not self.log_file.exists():
            return ""
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"Error reading log file: {e}")
            return ""
    
    def load_frame_images(self):
        """Load all images from the event_frames folder as base64."""
        if not self.frames_folder.exists():
            return []
        
        frames_base64 = []
        image_files = sorted(self.frames_folder.glob("*.jpg"))

        print(f"Found {len(image_files)} image file(s) in {self.frames_folder}")
        
        for img_path in image_files:
            try:
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
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        return frames_base64
    
    def summarize_with_openai(self, log_content, frames_base64, model="gpt-5"):
        """
        Send log content and frames to OpenAI API for summarization.
        
        Args:
            log_content: Text content from log file
            frames_base64: List of dicts with 'filename' and 'data' keys
            model: OpenAI model to use
        
        Returns:
            Summary text from OpenAI
        """
        if not log_content and not frames_base64:
            return "No events or frames to summarize."
        
        client = OpenAI(api_key=self.api_key)

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
        for frame_info in frames_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_info['data']}",
                    "detail": "low"  # Use "low" for speed, "high" for detail
                }
            })
        
        print(f"Sending request to OpenAI: {len(frames_base64)} frames...")
        
        try:
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
            
            summary = response.choices[0].message.content
            return summary
        
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None
    
    def clear_log_and_frames(self):
        """Clear the log file and delete all frames in the folder."""
        # Clear log file
        if self.log_file.exists():
            try:
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    f.write("")
                print(f"Cleared {self.log_file}")
            except Exception as e:
                print(f"Error clearing log file: {e}")
        
        # Delete all frame images
        if self.frames_folder.exists():
            deleted_count = 0
            for img_path in self.frames_folder.glob("*.jpg"):
                try:
                    img_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {img_path}: {e}")
            
            if deleted_count > 0:
                print(f"Deleted {deleted_count} frame(s) from {self.frames_folder}")
    
    def run_summary_cycle(self):
        """Run one cycle of summary generation."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*60}")
        print(f"Running summary cycle at {timestamp}")
        print(f"{'='*60}")
        
        # Read log content
        log_content = self.read_log_content()
        print(f"Log content: {len(log_content)} characters")
        
        # Load frames
        frames = self.load_frame_images()
        print(f"Loaded {len(frames)} frame(s)")
        
        # Clear log and frames
        self.clear_log_and_frames()

        # Generate summary if there's content
        if log_content or frames:
            summary = self.summarize_with_openai(log_content, frames)
            
            if summary:
                print(f"\n{'='*60}")
                print("GAMEPLAY SUMMARY")
                print(f"{'='*60}")
                print(summary)
                print(f"{'='*60}\n")
                
                # Save summary to file
                summary_file = Path("summaries") / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                summary_file.parent.mkdir(exist_ok=True)
                
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Frames: {len(frames)}\n")
                    f.write(f"Log entries: {len(log_content.splitlines())}\n\n")
                    f.write(summary)
                
                print(f"Summary saved to: {summary_file}")
        else:
            print("No events or frames to summarize in this cycle.")
    
    def run(self):
        """Main loop: start event capture and run summaries periodically."""
        self.running = True
        
        self.clear_log_and_frames()

        # Start event capture
        self.start_event_capture()
        
        try:
            while self.running:
                time.sleep(self.interval)
                self.run_summary_cycle()
        
        except KeyboardInterrupt:
            print("\n\nStopping summarizer...")
        
        finally:
            self.stop_event_capture()
            print("Summarizer stopped.")

def main():
    # Configuration
    LOG_FILE = "log_test.txt"
    FRAMES_FOLDER = "event_frames"
    API_KEY = os.getenv("OPENAI_API_KEY")
    INTERVAL = 10  # seconds between summaries
    
    if not API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Check if event_cap.py exists
    if not Path("event_cap.py").exists():
        print("Error: event_cap.py not found in current directory")
        sys.exit(1)
    
    print("="*60)
    print("APEX LEGENDS EVENT SUMMARIZER")
    print("="*60)
    print(f"Log file: {LOG_FILE}")
    print(f"Frames folder: {FRAMES_FOLDER}")
    print(f"Summary interval: {INTERVAL} seconds")
    print("Press Ctrl+C to stop")
    print("="*60)
    
    # Create and run summarizer
    summarizer = EventSummarizer(
        log_file=LOG_FILE,
        frames_folder=FRAMES_FOLDER,
        api_key=API_KEY,
        interval=INTERVAL
    )
    
    summarizer.run()

if __name__ == "__main__":
    main()