import json
import time
from pathlib import Path
from datetime import datetime
import pyautogui
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def create_event_frames_folder():
    """Create the event_frames folder if it doesn't exist."""
    folder = Path("event_frames")
    folder.mkdir(exist_ok=True)
    return folder

def should_take_screenshot(json_obj):
    """
    Check if the JSON object contains any trigger events or keys.
    
    Triggers:
    - events with name: knocked_out, kill, assist, damage, healed_from_ko, respawn
    - match_info with key: tabs or teammate_# (where # is 0-2)
    """
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
    
    return False, None

def take_screenshot(folder, event_type):
    """Take a screenshot and save it with timestamp and event type."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
    filename = f"{timestamp}_{event_type}.jpg"
    filepath = folder / filename
    
    try:
        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)
        print(f"Screenshot saved: {filename}")
        return True
    except Exception as e:
        print(f"Error taking screenshot: {e}")
        return False

class LogFileHandler(FileSystemEventHandler):
    """Handler for file system events on the log file."""
    
    def __init__(self, filepath, output_folder):
        self.filepath = Path(filepath).resolve()
        self.output_folder = output_folder
        self.file_handle = None
        self.last_position = 0
        
        # Open file and move to end
        self._open_file()
    
    def _open_file(self):
        """Open the file and seek to the end."""
        if self.file_handle:
            self.file_handle.close()
        
        self.file_handle = open(self.filepath, 'r', encoding='utf-8')
        self.file_handle.seek(0, 2)  # Move to end
        self.last_position = self.file_handle.tell()
    
    def on_modified(self, event):
        """Called when the file is modified."""
        if Path(event.src_path).resolve() != self.filepath:
            return
        
        try:
            # Read new lines
            self.file_handle.seek(self.last_position)
            new_lines = self.file_handle.readlines()
            self.last_position = self.file_handle.tell()
            
            # Process each new line
            for line in new_lines:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        json_obj = json.loads(line)
                        should_capture, event_type = should_take_screenshot(json_obj)
                        
                        if should_capture:
                            take_screenshot(self.output_folder, event_type)
                    
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse JSON: {e}")
        
        except Exception as e:
            print(f"Error reading file: {e}")
            # Try to reopen the file
            self._open_file()
    
    def close(self):
        """Close the file handle."""
        if self.file_handle:
            self.file_handle.close()

def monitor_file(filepath):
    """
    Monitor a file for changes and take screenshots on trigger events.
    
    Args:
        filepath: Path to the text file to monitor
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"Error: File {filepath} does not exist!")
        return
    
    # Create output folder
    output_folder = create_event_frames_folder()
    print(f"Monitoring {filepath} for events...")
    print(f"Screenshots will be saved to: {output_folder.absolute()}")
    print("Press Ctrl+C to stop monitoring\n")
    
    # Create event handler and observer
    event_handler = LogFileHandler(filepath, output_folder)
    observer = Observer()
    
    # Watch the directory containing the file
    observer.schedule(event_handler, str(filepath.parent), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
    finally:
        observer.stop()
        observer.join()
        event_handler.close()

if __name__ == "__main__":
    import sys
    
    # Check if required libraries are installed
    try:
        import pyautogui
    except ImportError:
        print("Error: pyautogui is not installed.")
        print("Install it with: pip install pyautogui")
        sys.exit(1)
    
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("Error: watchdog is not installed.")
        print("Install it with: pip install watchdog")
        sys.exit(1)
    
    # Get filename from command line or use default
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Default to test_apex.txt
        filename = "log_test.txt"
    
    monitor_file(filename)