# Collaborative Air Canvas

A real-time collaborative air drawing application that allows multiple users to draw together in the air using hand gestures. The application uses computer vision to track hand movements and integrates with Google's Gemini AI to convert sketches into professional diagrams.

## Features

### ✨ Core Features
- **Air Drawing**: Draw in the air using hand gestures detected by your webcam
- **Real-time Collaboration**: Multiple users can draw together simultaneously
- **AI-Powered Diagram Generation**: Convert sketches to Mermaid/PlantUML diagrams using Gemini AI
- **Professional Tools**: Freehand, Line, Rectangle, Circle, and Eraser tools
- **Color Palette**: Multiple colors for different users and drawing needs
- **Live Preview**: See shapes as you draw them before finalizing

### 🤝 Collaborative Features
- **Multi-user Support**: Connect multiple users to the same drawing session
- **Real-time Synchronization**: All drawing actions are synchronized instantly
- **User Identification**: Each user gets a unique color and name
- **Shared Canvas**: Everyone works on the same canvas
- **Collaborative AI**: Generated diagrams are shared with all users

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Internet connection (for Gemini AI features)

### Step 1: Clone and Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Gemini API (Optional)
1. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Replace the API key in both `collaborative_airCanvas.py` and `websocket_server.py`:
```python
GEMINI_API_KEY = 'YOUR_API_KEY_HERE'
```

## Usage

### Running the Collaborative Application

#### Step 1: Start the Server
```bash
python websocket_server.py
```
You should see: `Starting collaborative canvas server on ws://localhost:8765`

#### Step 2: Start Client(s)
In separate terminal windows (or on different computers):
```bash
python collaborative_airCanvas.py
```

Each client will connect to the server and be assigned a unique username and color.

### How to Draw

1. **Position your hand** in front of the webcam
2. **Point with your index finger** - this is your drawing cursor
3. **Keep your middle finger down** while drawing to activate drawing mode
4. **Select tools and colors** from the toolbar by pointing at them
5. **Draw in the air** - your movements will appear on the canvas in real-time

### Gestures Guide

| Gesture | Action |
|---------|--------|
| Index finger up, middle down | Drawing mode |
| Point at toolbar items | Select colors/tools |
| Release gesture | Stop drawing/finalize shapes |

### Tools Available

- **Freehand**: Draw freely with smooth lines
- **Line**: Draw straight lines
- **Rectangle**: Draw rectangles
- **Circle**: Draw circles
- **Eraser**: Erase parts of the drawing

### Collaborative Features

- **Multiple Users**: Each user appears with their own color
- **Real-time Updates**: See other users' drawings instantly
- **Shared Actions**: Clear and save actions affect all users
- **AI Sharing**: Generated diagrams are shared with everyone

## AI Diagram Generation

When you click "Save", the application:
1. Saves your drawing locally
2. Sends it to Gemini AI for analysis
3. Generates structured diagram code (Mermaid/PlantUML)
4. Displays the generated diagram
5. Shares the result with all connected users

### Supported Diagram Types
- Flowcharts
- UML diagrams
- System architecture diagrams
- Mind maps
- Process flows

### Components
1. **WebSocket Server** (`websocket_server.py`): Handles real-time communication
2. **Collaborative Client** (`collaborative_airCanvas.py`): Main application with collaboration
3. **Original Client** (`airCanvas.py`): Single-user version
