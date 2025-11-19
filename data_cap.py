import os
import time
import subprocess
import signal
import sys
import shutil
import threading
import queue
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from mss import mss
import json
import pyautogui
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class LogFileHandler(FileSystemEventHandler):
    """Handler for file system events on the log file."""
    
    def __init__(self, filepath, frames_folder):
        self.filepath = Path(filepath).resolve()
        self.frames_folder = Path(frames_folder)
        self.file_handle = None
        self.last_position = 0
        
        # Create frames folder if it doesn't exist
        self.frames_folder.mkdir(exist_ok=True)
        
        # Open file and move to end
        self._open_file()
    
    def _open_file(self):
        """Open the file and seek to the end."""
        if self.file_handle:
            self.file_handle.close()
        
        # Ensure file exists
        if not self.filepath.exists():
            self.filepath.touch()
        
        self.file_handle = open(self.filepath, 'r', encoding='utf-8')
        self.file_handle.seek(0, 2)  # Move to end
        self.last_position = self.file_handle.tell()
    
    def should_take_screenshot(self, json_obj):
        """Check if the JSON object contains any trigger events or keys."""
        # Check for events
        if "events" in json_obj:
            for event in json_obj["events"]:
                if "name" in event:
                    trigger_events = ["knocked_out", "kill", "assist", "damage", 
                                    "healed_from_ko", "respawn"]
                    if event["name"] in trigger_events:
                        return True, f"event_{event['name']}"
        
        # Check for match_info with specific keys
        if "match_info" in json_obj:
            match_info = json_obj["match_info"]
            
            # Check for tabs key
            if "tabs" in match_info:
                return True, "match_info_tabs"
            
            # Check for teammate_0, teammate_1, teammate_2
            for i in range(3):
                teammate_key = f"teammate_{i}"
                if teammate_key in match_info:
                    return True, f"match_info_{teammate_key}"
                
        if "me" in json_obj:
            me = json_obj["me"]

            if "inUse" in me:
                return True, "inUse"

        return False, None
    
    def take_screenshot(self, event_type):
        """Take a screenshot and save it with timestamp and event type."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{timestamp}_{event_type}.jpg"
        filepath = self.frames_folder / filename
        
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            print(f"[EventCapture] Screenshot: {filename}")
            return True
        except Exception as e:
            print(f"[EventCapture] Error taking screenshot: {e}")
            return False
    
    def on_modified(self, event):
        """Called when the file is modified."""
        if Path(event.src_path).resolve() != self.filepath:
            return
        
        try:
            # Get current file size
            current_size = self.filepath.stat().st_size
            
            # Check if file was truncated (cleared)
            if current_size < self.last_position:
                print(f"[EventCapture] File was truncated, resetting position")
                self.last_position = 0
                self.file_handle.seek(0)
            else:
                # Seek to last position
                self.file_handle.seek(self.last_position)

            #Read new lines    
            new_lines = self.file_handle.readlines()
            self.last_position = self.file_handle.tell()
            
            # Process each new line in parallel using threads
            threads = []
            for line in new_lines:
                line = line.strip()
                if line:  # Skip empty lines
                    # Process each line in a separate thread for true parallelism
                    # thread = threading.Thread(
                    #     target=self._process_line,
                    #     args=(line,),
                    #     daemon=True
                    # )
                    # thread.start()
                    # threads.append(thread)
                    try:
                        json_obj = json.loads(line)
                        should_capture, event_type = self.should_take_screenshot(json_obj)
                        
                        if should_capture:
                            self.take_screenshot(event_type)
                    
                    except json.JSONDecodeError as e:
                        print(f"[EventCapture] Warning: Could not parse JSON: {e}")
            
            # Wait for all screenshot threads to complete
            # for thread in threads:
            #     thread.join(timeout=1)
        
        except Exception as e:
            print(f"[EventCapture] Error reading file: {e}")
            # Try to reopen the file
            self._open_file()
    
    def _process_line(self, line):
        """Process a single log line (runs in separate thread)."""
        try:
            json_obj = json.loads(line)
            should_capture, event_type = self.should_take_screenshot(json_obj)
            
            if should_capture:
                self.take_screenshot(event_type)
        
        except json.JSONDecodeError as e:
            print(f"[EventCapture] Warning: Could not parse JSON: {e}")
        except Exception as e:
            print(f"[EventCapture] Error processing line: {e}")
    
    def close(self):
        """Close the file handle."""
        if self.file_handle:
            self.file_handle.close()


class EventCaptureThread(threading.Thread):
    """Thread to run event capture with watchdog observer."""
    
    def __init__(self, log_file, frames_folder):
        super().__init__(daemon=True)
        self.log_file = Path(log_file)
        self.frames_folder = Path(frames_folder)
        self.running = False
        self.observer = None
        self.event_handler = None
    
    def run(self):
        """Main thread loop."""
        print("[EventCapture] Thread started with watchdog observer")
        self.running = True
        
        # Create event handler and observer
        self.event_handler = LogFileHandler(self.log_file, self.frames_folder)
        self.observer = Observer()
        
        # Watch the directory containing the file
        self.observer.schedule(
            self.event_handler, 
            str(self.log_file.parent), 
            recursive=False
        )
        self.observer.start()
        
        try:
            while self.running:
                time.sleep(0.1)
        
        except Exception as e:
            print(f"[EventCapture] Thread error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            if self.event_handler:
                self.event_handler.close()
            print("[EventCapture] Thread stopped")
    
    def stop(self):
        """Stop the thread."""
        print("[EventCapture] Stopping thread...")
        self.running = False


class EventDataCollector:
    def __init__(self, log_file="log_test.txt", frames_folder="event_frames", 
                 output_folder="sample_data", interval=10, monitor_number=0):
        """
        Initialize the event data collector.
        
        Args:
            log_file: Path to the log text file
            frames_folder: Path to the event frames folder
            output_folder: Path to the sample data output folder
            interval: Time in seconds between collections (also video duration)
            monitor_number: Monitor to capture (0 for all, 1 for primary, 2+ for other monitors)
        """
        self.log_file = Path(log_file)
        self.frames_folder = Path(frames_folder)
        self.output_folder = Path(output_folder)
        self.interval = interval
        self.monitor_number = monitor_number
        self.event_capture_thread = None
        self.running = False
        self.collection_count = 0
        self.recording_active = False
        self.current_video_path = None
        self.video_writer = None
        self.video_thread = None
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(exist_ok=True)
    
    def start_event_capture(self):
        """Start the event capture thread."""
        print("Starting event capture thread...")
        self.event_capture_thread = EventCaptureThread(
            self.log_file,
            self.frames_folder
        )
        self.event_capture_thread.start()
        print("Event capture thread started.")
    
    def stop_event_capture(self):
        """Stop the event capture thread."""
        if self.event_capture_thread:
            print("\nStopping event capture thread...")
            self.event_capture_thread.stop()
            self.event_capture_thread.join(timeout=2)
            print("Event capture thread stopped.")
    
    def start_video_recording(self, destination_folder):
        """
        Start continuous video recording to a file.
        
        Args:
            destination_folder: Path to save the video
        """
        self.current_video_path = destination_folder / "gameplay_recording.mp4"
        self.recording_active = True
        self.captured_frames = []
        self.frame_timestamps = []
        
        # Start recording in a separate thread
        self.video_thread = threading.Thread(
            target=self._record_video_loop,
            daemon=True
        )
        self.video_thread.start()
        print(f"Started video recording to: {self.current_video_path}")
    
    def _record_video_loop(self):
        """Internal method to continuously record video frames to memory."""
        target_fps = 30.0
        frame_time = 1.0 / target_fps
        
        try:
            with mss() as sct:
                # Get the specified monitor
                monitors = sct.monitors
                print(f"Available monitors: {len(monitors) - 1}")
                for i, mon in enumerate(monitors):
                    print(f"  Monitor {i}: {mon}")
                
                # Use specified monitor
                if self.monitor_number >= len(monitors):
                    print(f"Warning: Monitor {self.monitor_number} not found, using monitor 0 (all)")
                    monitor = monitors[0]
                else:
                    monitor = monitors[self.monitor_number]
                
                print(f"Recording from monitor {self.monitor_number}: {monitor['width']}x{monitor['height']}")
                
                frame_count = 0
                start_time = time.time()
                next_frame_time = start_time
                
                while self.recording_active:
                    current_time = time.time()
                    
                    # Only capture if it's time for the next frame
                    if current_time >= next_frame_time:
                        # Capture screen
                        screenshot = sct.grab(monitor)
                        
                        # Convert to numpy array
                        frame = np.array(screenshot)
                        
                        # Convert BGRA to BGR (remove alpha channel)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        
                        # Store frame and timestamp
                        self.captured_frames.append(frame)
                        self.frame_timestamps.append(current_time - start_time)
                        frame_count += 1
                        
                        # Schedule next frame
                        next_frame_time += frame_time
                    else:
                        # Sleep briefly to avoid busy waiting
                        sleep_time = min(0.001, next_frame_time - current_time)
                        time.sleep(sleep_time)
                
                duration = time.time() - start_time
                actual_fps = frame_count / duration if duration > 0 else 0
                
                print(f"Captured {frame_count} frames in {duration:.2f}s, {actual_fps:.1f} actual fps")
                
                # Now write the video with the ACTUAL fps
                self._write_video_with_actual_fps(monitor['width'], monitor['height'], actual_fps)
        
        except Exception as e:
            print(f"Error during video recording: {e}")
            import traceback
            traceback.print_exc()
    
    def _write_video_with_actual_fps(self, width, height, actual_fps):
        """Write captured frames to video file with actual achieved FPS."""
        if not self.captured_frames:
            print("No frames to write")
            return
        
        try:
            print(f"Writing video with {actual_fps:.2f} fps ({len(self.captured_frames)} frames)...")
            
            # Use the actual FPS we achieved for encoding
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(self.current_video_path),
                fourcc,
                actual_fps,  # Use actual captured FPS
                (width, height)
            )
            
            if not video_writer.isOpened():
                print("Error: Could not open video writer")
                return
            
            # Write all frames
            for frame in self.captured_frames:
                video_writer.write(frame)
            
            video_writer.release()
            print(f"Video written successfully: {self.current_video_path}")
            
        except Exception as e:
            print(f"Error writing video: {e}")
            import traceback
            traceback.print_exc()
    
    def stop_video_recording(self):
        """Stop the current video recording."""
        if self.recording_active:
            print("Stopping video recording...")
            self.recording_active = False
            
            # Wait for video thread to finish
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=5)
            
            print("Video recording stopped.")
    
    def copy_log_file(self, destination_folder):
        """
        Copy the log file to the destination folder.
        
        Args:
            destination_folder: Path to copy the log file to
        
        Returns:
            Tuple of (success, line_count)
        """
        if not self.log_file.exists():
            print(f"Log file not found: {self.log_file}")
            return False, 0
        
        try:
            # Read and count lines
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            line_count = len([line for line in lines if line.strip()])
            file_size = self.log_file.stat().st_size
            
            if file_size == 0:
                print(f"Log file is empty: {self.log_file}")
                return False, 0
            
            dest_path = destination_folder / self.log_file.name
            shutil.copy2(self.log_file, dest_path)
            print(f"Copied log file: {line_count} lines, {file_size} bytes")
            return True, line_count
        
        except Exception as e:
            print(f"Error copying log file: {e}")
            return False, 0
    
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
        # Look for both PNG and JPG files
        image_files = list(self.frames_folder.glob("*.jpg"))
        image_files = sorted(image_files)
        
        print(f"Found {len(image_files)} frame(s) to copy")
        
        for img_path in image_files:
            try:
                dest_path = frames_dest / img_path.name
                shutil.copy2(img_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {img_path.name}: {e}")
        
        if copied_count > 0:
            print(f"Copied {copied_count} frame(s)")
        
        return copied_count
    
    def clear_log_and_frames(self):
        """Clear the log file and delete all frames in the folder."""
        # Clear log file
        if self.log_file.exists():
            try:
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    f.write("")
                print(f"Cleared log file: {self.log_file}")
            except Exception as e:
                print(f"Error clearing log file: {e}")
        
        # Delete all frame images (both PNG and JPG)
        if self.frames_folder.exists():
            deleted_count = 0
            # for ext in ['*.png', '*.jpg']:
            for img_path in self.frames_folder.glob('*.jpg'):
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
        
        # Start video recording for this interval
        self.start_video_recording(collection_folder)
        
        # Wait for the interval duration
        print(f"Recording for {self.interval} seconds...")
        time.sleep(self.interval)
        
        # Stop video recording
        self.stop_video_recording()
        
        # Small delay to ensure all files are written
        # time.sleep(0.5)
        
        # Copy log file
        log_copied, log_lines = self.copy_log_file(collection_folder)
        
        # Copy frames
        frames_copied = self.copy_frame_images(collection_folder)
        
        # Clear log and frames for next cycle
        print("\nClearing log and frames for next cycle...")
        self.clear_log_and_frames()
        
        # Create metadata file
        metadata_path = collection_folder / "metadata.txt"
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(f"Collection Time: {timestamp}\n")
                f.write(f"Collection Number: {self.collection_count}\n")
                f.write(f"Recording Duration: {self.interval} seconds\n")
                f.write(f"Monitor: {self.monitor_number}\n")
                f.write(f"Log File Copied: {log_copied}\n")
                f.write(f"Log Lines: {log_lines}\n")
                f.write(f"Frames Copied: {frames_copied}\n")
                f.write(f"Video File: gameplay_recording.mp4\n")
            print(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            print(f"Error writing metadata: {e}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Collection #{self.collection_count} Summary:")
        print(f"  Video: ✓ Recorded ({self.interval}s on monitor {self.monitor_number})")
        print(f"  Log: {'✓ Copied' if log_copied else '✗ Not copied'} ({log_lines} lines)")
        print(f"  Frames: {frames_copied} copied")
        print(f"  Cleared: ✓ Ready for next cycle")
        print(f"  Location: {collection_folder}")
        print(f"{'='*60}")
    
    def run(self):
        """Main loop: start event capture and collect data periodically."""
        self.running = True
        
        # Clear any existing data before starting
        print("Clearing any existing log and frames...")
        self.clear_log_and_frames()
        
        # Start event capture
        self.start_event_capture()
        
        # Give event capture time to start
        time.sleep(1)
        
        print(f"\nData collection starting...")
        print(f"Interval: {self.interval} seconds")
        print(f"Monitor: {self.monitor_number}")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                self.run_collection_cycle()
        
        except KeyboardInterrupt:
            print("\n\nStopping data collector...")
        
        finally:
            # Stop any active recording
            if self.recording_active:
                self.stop_video_recording()
            
            self.stop_event_capture()
            print(f"\nData collector stopped. Total collections: {self.collection_count}")

def main():
    # Configuration
    LOG_FILE = "log_test.txt"
    FRAMES_FOLDER = "event_frames"
    OUTPUT_FOLDER = "sample_data"
    INTERVAL = 10  # seconds between collections
    MONITOR = 2  # 0 = all monitors, 1 = primary, 2 = secondary, etc.
    
    # Check if required libraries are installed
    try:
        import cv2
    except ImportError:
        print("Error: opencv-python is not installed.")
        print("Install it with: pip install opencv-python")
        sys.exit(1)
    
    try:
        import mss
    except ImportError:
        print("Error: mss is not installed.")
        print("Install it with: pip install mss")
        sys.exit(1)
    
    try:
        import pyautogui
    except ImportError:
        print("Error: pyautogui is not installed.")
        print("Install it with: pip install pyautogui")
        sys.exit(1)
    
    print("="*60)
    print("APEX LEGENDS EVENT DATA COLLECTOR")
    print("="*60)
    print(f"Log file: {LOG_FILE}")
    print(f"Frames folder: {FRAMES_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Collection interval: {INTERVAL} seconds")
    print(f"Monitor: {MONITOR}")
    print("="*60)
    
    # Create and run collector
    collector = EventDataCollector(
        log_file=LOG_FILE,
        frames_folder=FRAMES_FOLDER,
        output_folder=OUTPUT_FOLDER,
        interval=INTERVAL,
        monitor_number=MONITOR
    )
    
    collector.run()

if __name__ == "__main__":
    main()