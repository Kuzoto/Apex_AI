// Constants
const APEX_GAME_ID = 21566;
const PYTHON_SERVER_URL = 'http://localhost:5000/events'; // Your Python server endpoint

// Event tracking
let eventLog = [];
let isGameRunning = false;

// Initialize the app
function initialize() {
  console.log('Apex Events Tracker initialized');
  
  // Register game events listener
  overwolf.games.events.setRequiredFeatures(
    [
      'kill',
      'death',
      'assist',
      'headshot',
      'knockdown',
      'damage',
      'revive',
      'match_state',
      'match_info',
      'inventory',
      'location',
      'rank',
      'me',
      'team',
      'roster'
    ],
    (result) => {
      if (result.success) {
        console.log('Required features set successfully:', result);
      } else {
        console.error('Failed to set required features:', result);
      }
    }
  );

  // Listen for game launch
  overwolf.games.onGameLaunched.addListener((event) => {
    if (event.id === APEX_GAME_ID) {
      console.log('Apex Legends launched');
      isGameRunning = true;
      sendEventToPython({ type: 'game_launched', timestamp: Date.now() });
      openInGameWindow();
    }
  });

  // Listen for game exit
  overwolf.games.onGameInfoUpdated.addListener((event) => {
    if (event.gameInfo && event.gameInfo.isRunning === false && isGameRunning) {
      console.log('Apex Legends closed');
      isGameRunning = false;
      sendEventToPython({ type: 'game_closed', timestamp: Date.now() });
      closeInGameWindow();
    }
  });

  // Listen for new game events
  overwolf.games.events.onNewEvents.addListener((events) => {
    console.log('New events:', events);
    if (events && events.events) {
      events.events.forEach((event) => {
        handleGameEvent(event);
      });
    }
  });

  // Listen for info updates (game state changes)
  overwolf.games.events.onInfoUpdates2.addListener((info) => {
    console.log('Info updates:', info);
    if (info && info.info) {
      handleInfoUpdate(info.info);
    }
  });

  // Check if game is already running
  overwolf.games.getRunningGameInfo((gameInfo) => {
    if (gameInfo && gameInfo.id === APEX_GAME_ID && gameInfo.isRunning) {
      isGameRunning = true;
      openInGameWindow();
    }
  });
}

// Handle game events
function handleGameEvent(event) {
  const eventData = {
    type: 'game_event',
    name: event.name,
    data: event.data,
    timestamp: Date.now()
  };
  
  eventLog.push(eventData);
  console.log('Game event:', eventData);
  
  // Send to Python
  sendEventToPython(eventData);
}

// Handle info updates
function handleInfoUpdate(info) {
  const updateData = {
    type: 'info_update',
    data: info,
    timestamp: Date.now()
  };
  
  console.log('Info update:', updateData);
  
  // Send to Python
  sendEventToPython(updateData);
}

// Send event to Python server
async function sendEventToPython(eventData) {
  try {
    const response = await fetch(PYTHON_SERVER_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(eventData)
    });
    
    if (!response.ok) {
      console.error('Failed to send event to Python:', response.status);
    } else {
      console.log('Event sent successfully');
    }
  } catch (error) {
    console.error('Error sending event to Python:', error);
    // Store event for retry if server is down
    // You could implement a retry queue here
  }
}

// Window management
function openInGameWindow() {
  overwolf.windows.obtainDeclaredWindow('in_game', (result) => {
    if (result.success) {
      overwolf.windows.restore(result.window.id, () => {
        console.log('In-game window opened');
      });
    }
  });
}

function closeInGameWindow() {
  overwolf.windows.obtainDeclaredWindow('in_game', (result) => {
    if (result.success) {
      overwolf.windows.close(result.window.id);
    }
  });
}

// Get event log (for display in UI)
function getEventLog() {
  return eventLog;
}

// Initialize when script loads
initialize();