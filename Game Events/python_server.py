from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for Overwolf app

# Storage for events
events_log = []
EVENT_LOG_FILE = 'apex_events.json'

# Load existing events if file exists
if os.path.exists(EVENT_LOG_FILE):
    try:
        with open(EVENT_LOG_FILE, 'r') as f:
            events_log = json.load(f)
        print(f"Loaded {len(events_log)} existing events from {EVENT_LOG_FILE}")
    except Exception as e:
        print(f"Error loading events: {e}")

@app.route('/events', methods=['POST'])
def receive_event():
    """Receive game events from Overwolf app"""
    try:
        event_data = request.get_json()
        
        if not event_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Add server timestamp
        event_data['server_timestamp'] = datetime.now().isoformat()
        
        # Store event
        events_log.append(event_data)
        
        # Log to console
        print(f"\n{'='*60}")
        print(f"Event Type: {event_data.get('type')}")
        if event_data.get('name'):
            print(f"Event Name: {event_data.get('name')}")
        print(f"Timestamp: {event_data.get('timestamp')}")
        print(f"Data: {json.dumps(event_data.get('data', {}), indent=2)}")
        print(f"{'='*60}\n")
        
        # Process specific event types
        process_event(event_data)
        
        # Save to file periodically (every 10 events)
        if len(events_log) % 10 == 0:
            save_events()
        
        return jsonify({'success': True, 'message': 'Event received'}), 200
        
    except Exception as e:
        print(f"Error processing event: {e}")
        return jsonify({'error': str(e)}), 500

def process_event(event):
    """Process specific event types"""
    event_type = event.get('type')
    event_name = event.get('name', '')
    
    if event_type == 'game_launched':
        print("🎮 Apex Legends has started!")
        
    elif event_type == 'game_closed':
        print("👋 Apex Legends has closed!")
        save_events()  # Save all events on game close
        
    elif event_type == 'game_event':
        if event_name == 'kill':
            print("💀 KILL EVENT!")
        elif event_name == 'death':
            print("☠️ DEATH EVENT!")
        elif event_name == 'knockdown':
            print("🎯 KNOCKDOWN!")
        elif event_name == 'damage':
            data = event.get('data', {})
            print(f"⚔️ DAMAGE: {data}")
        elif event_name == 'headshot':
            print("🎯 HEADSHOT!")
            
    elif event_type == 'info_update':
        data = event.get('data', {})
        # You can add specific handling for info updates here
        if 'match_state' in data:
            print(f"🎮 Match State: {data['match_state']}")

def save_events():
    """Save events to file"""
    try:
        with open(EVENT_LOG_FILE, 'w') as f:
            json.dump(events_log, f, indent=2)
        print(f"✅ Saved {len(events_log)} events to {EVENT_LOG_FILE}")
    except Exception as e:
        print(f"❌ Error saving events: {e}")

@app.route('/events', methods=['GET'])
def get_events():
    """Get all stored events"""
    return jsonify({
        'total_events': len(events_log),
        'events': events_log
    }), 200

@app.route('/events/stats', methods=['GET'])
def get_stats():
    """Get statistics about events"""
    stats = {
        'total_events': len(events_log),
        'event_types': {},
        'game_events': {}
    }
    
    for event in events_log:
        event_type = event.get('type')
        stats['event_types'][event_type] = stats['event_types'].get(event_type, 0) + 1
        
        if event_type == 'game_event':
            event_name = event.get('name')
            stats['game_events'][event_name] = stats['game_events'].get(event_name, 0) + 1
    
    return jsonify(stats), 200

@app.route('/events/clear', methods=['POST'])
def clear_events():
    """Clear all stored events"""
    global events_log
    events_log = []
    if os.path.exists(EVENT_LOG_FILE):
        os.remove(EVENT_LOG_FILE)
    return jsonify({'success': True, 'message': 'Events cleared'}), 200

if __name__ == '__main__':
    print("🚀 Starting Apex Events Python Server...")
    print("📡 Listening on http://localhost:5000")
    print("📊 Endpoints:")
    print("   POST /events - Receive events from Overwolf")
    print("   GET  /events - Get all events")
    print("   GET  /events/stats - Get event statistics")
    print("   POST /events/clear - Clear all events")
    print("\n⌚ Waiting for events...\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)