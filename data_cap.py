import os
import time
import subprocess
import signal
import sys
import shutil
import threading
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from mss import mss

class EventDataCollector:
    def __init__(self, log_file="log_test.txt", frames_folder="event_frames", 
                 output_folder="sample_data", interval=10):
        """
        Initialize the event data collector.
        
        Args:
            log_file: Path to the log text file
            frames_folder: Path to the event frames folder
            output_folder: Path to the sample data output folder
            interval: Time in seconds between collections (also video duration)
        """
        self.log_file = Path(log_file)
        self.frames_folder = Path(frames_folder)
        self.output_folder = Path(output_folder)
        self.interval = interval
        self.event_cap_process = None
        self.running = False
        self.collection_count = 0
        self.video_recording = False
        self.video_frames = []
        self.video_lock = threading.Lock()
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(exist_ok=True)
    
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
    
    def record_screen(self, duration, destination_folder):
        """
        Record screen for specified duration and save as video.
        
        Args:
            duration: Recording duration in seconds
            destination_folder: Path to save the video
        
        Returns:
            Path to saved video file, or None if failed
        """
        print(f"Starting screen recording for {duration} seconds...")
        
        video_filename = destination_folder / "gameplay_recording.mp4"
        fps = 30  # Frames per second
        
        try:
            with mss() as sct:
                # Get the primary monitor
                monitor = sct.monitors[1]
                
                # Prepare video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                width = monitor["width"]
                height = monitor["height"]
                out = cv2.VideoWriter(str(video_filename), fourcc, fps, (width, height))
                
                if not out.isOpened():
                    print("Error: Could not open video writer")
                    return None
                
                start_time = time.time()
                frame_count = 0
                
                while time.time() - start_time < duration:
                    # Capture screen
                    screenshot = sct.grab(monitor)
                    
                    # Convert to numpy array
                    frame = np.array(screenshot)
                    
                    # Convert BGRA to BGR (remove alpha channel)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    
                    # Write frame
                    out.write(frame)
                    frame_count += 1
                    
                    # Control frame rate
                    time.sleep(1.0 / fps)
                
                out.release()
                
                print(f"Screen recording complete: {frame_count} frames saved to {video_filename}")
                return video_filename
        
        except Exception as e:
            print(f"Error during screen recording: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def copy_log_file(self, destination_folder):
        """
        Copy the log file to the destination folder.
        
        Args:
            destination_folder: Path to copy the log file to
        
        Returns:
            True if successful, False otherwise
        """
        if not self.log_file.exists():
            print(f"Log file not found: {self.log_file}")
            return False
        
        try:
            # Check if file has content
            file_size = self.log_file.stat().st_size
            if file_size == 0:
                print(f"Log file is empty: {self.log_file}")
                return False
            
            dest_path = destination_folder / self.log_file.name
            shutil.copy2(self.log_file, dest_path)
            print(f"Copied log file to: {dest_path} ({file_size} bytes)")
            return True
        
        except Exception as e:
            print(f"Error copying log file: {e}")
            return False
    
    def copy_frame_images(self, destination_folder):
        """
        Copy all frame images to the destination folder.
        
        Args:
            destination_folder: Path to copy the frames to
        
        Returns:
            Number of frames copied
        """
        if not self.frames_folder.exists():
            print(f"Frames folder not found: {self.frames_folder}")
            return 0
        
        # Create frames subfolder in destination
        frames_dest = destination_folder / "frames"
        frames_dest.mkdir(exist_ok=True)
        
        copied_count = 0
        image_files = sorted(self.frames_folder.glob("*.jpg"))
        
        print(f"Found {len(image_files)} frame(s) to copy")
        
        for img_path in image_files:
            try:
                dest_path = frames_dest / img_path.name
                shutil.copy2(img_path, dest_path)
                print(f"Copied frame: {img_path.name}")
                copied_count += 1
            except Exception as e:
                print(f"Error copying {img_path.name}: {e}")
        
        return copied_count
    
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
    
    def run_collection_cycle(self):
        """Run one cycle of data collection."""
        self.collection_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_safe = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*60}")
        print(f"Collection cycle #{self.collection_count} at {timestamp}")
        print(f"{'='*60}")
        
        # Create timestamped folder for this collection
        collection_folder = self.output_folder / f"collection_{timestamp_safe}"
        collection_folder.mkdir(exist_ok=True)
        print(f"Collection folder: {collection_folder}")
        
        # Start screen recording in parallel
        video_thread = threading.Thread(
            target=self.record_screen,
            args=(self.interval, collection_folder)
        )
        video_thread.start()
        
        # Wait for recording to complete
        video_thread.join()
        
        # Copy log file
        log_copied = self.copy_log_file(collection_folder)
        
        # Copy frames
        frames_copied = self.copy_frame_images(collection_folder)

        self.clear_log_and_frames()
        
        # Create metadata file
        metadata_path = collection_folder / "metadata.txt"
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(f"Collection Time: {timestamp}\n")
                f.write(f"Collection Number: {self.collection_count}\n")
                f.write(f"Recording Duration: {self.interval} seconds\n")
                f.write(f"Log File Copied: {log_copied}\n")
                f.write(f"Frames Copied: {frames_copied}\n")
            print(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            print(f"Error writing metadata: {e}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Collection #{self.collection_count} Summary:")
        print(f"  Video: ✓ Recorded ({self.interval}s)")
        print(f"  Log file: {'✓ Copied' if log_copied else '✗ Not copied'}")
        print(f"  Frames: {frames_copied} copied")
        print(f"  Location: {collection_folder}")
        print(f"{'='*60}")
    
    def run(self):
        """Main loop: start event capture and collect data periodically."""
        self.running = True
        
        self.clear_log_and_frames()
        
        # Start event capture
        self.start_event_capture()
        
        print(f"\nData collection will run every {self.interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                self.run_collection_cycle()
        
        except KeyboardInterrupt:
            print("\n\nStopping data collector...")
        
        finally:
            self.stop_event_capture()
            print(f"\nData collector stopped. Total collections: {self.collection_count}")

def main():
    # Configuration
    LOG_FILE = "log_test.txt"
    FRAMES_FOLDER = "event_frames"
    OUTPUT_FOLDER = "sample_data"
    INTERVAL = 10  # seconds between collections
    
    # Check if event_cap.py exists
    if not Path("event_cap.py").exists():
        print("Error: event_cap.py not found in current directory")
        sys.exit(1)
    
    print("="*60)
    print("APEX LEGENDS EVENT DATA COLLECTOR")
    print("="*60)
    print(f"Log file: {LOG_FILE}")
    print(f"Frames folder: {FRAMES_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Collection interval: {INTERVAL} seconds")
    print("="*60)
    
    # Create and run collector
    collector = EventDataCollector(
        log_file=LOG_FILE,
        frames_folder=FRAMES_FOLDER,
        output_folder=OUTPUT_FOLDER,
        interval=INTERVAL
    )
    
    collector.run()

if __name__ == "__main__":
    main()