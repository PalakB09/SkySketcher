import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
from datetime import datetime
import google.generativeai as genai
import PIL.Image
import base64
import io
import requests
import re
import urllib.parse
import asyncio
import websockets
import json
import threading
from queue import Queue

# Configure Gemini API
GEMINI_API_KEY = 'YOUR_API_KEY'
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    GEMINI_AVAILABLE = True
    print("Gemini API configured successfully!")
except Exception as e:
    print(f"Gemini API not configured: {e}")
    GEMINI_AVAILABLE = False

# WebSocket connection variables
WEBSOCKET_URL = "ws://localhost:8765"
websocket_connection = None
client_id = None
user_name = None
user_color = None
connected_users = {}
message_queue = Queue()

# Network optimization settings
BATCH_SIZE = 5  # Batch drawing points before sending
SEND_INTERVAL = 0.05  # Send batched data every 50ms
drawing_batch = []
last_send_time = 0

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Layout settings
WHITEBOARD_WIDTH = 1200
WHITEBOARD_HEIGHT = 720
CAMERA_WIDTH = 280
CAMERA_HEIGHT = 210
CAMERA_Y_OFFSET = 100
TOTAL_WIDTH = WHITEBOARD_WIDTH + CAMERA_WIDTH

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Toolbar settings - Professional Design
toolbar_height = 140
toolbar_color = (45, 45, 50)
button_color = (70, 75, 85)
button_hover_color = (90, 95, 105)
text_color = (240, 240, 245)
highlight_color = (52, 152, 219)
save_button_color = (46, 204, 113)
clear_button_color = (231, 76, 60)
shadow_color = (20, 20, 25)

# Colors in BGR format
colors = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
]

# Tools available
tools = ["Freehand", "Line", "Rectangle", "Circle", "Eraser"]

# Initial settings
current_color_index = 0
current_tool_index = 0
brush_thickness = 5
eraser_thickness = 30
drawing_mode = False
smoothing_factor = 0.7  # Increased smoothing for better experience
save_button_clicked = False

# Points for shape drawing
start_x, start_y = 0, 0
preview_shape = None
shape_started = False

# Canvas to draw on (white whiteboard)
canvas = np.full((WHITEBOARD_HEIGHT, WHITEBOARD_WIDTH, 3), 255, dtype=np.uint8)

# Previous positions for smooth drawing
prev_points = []
MAX_POINTS = 8  # Increased for smoother drawing

# Gesture states
gesture_start_time = 0
GESTURE_DELAY = 0.08  # Reduced delay for more responsive drawing

# Create saves directory
if not os.path.exists('saved_drawings'):
    os.makedirs('saved_drawings')

# WebSocket functions
async def connect_to_server():
    """Connect to the WebSocket server"""
    global websocket_connection, client_id, user_name, user_color
    
    try:
        websocket_connection = await websockets.connect(WEBSOCKET_URL)
        print("Connected to collaborative server!")
        
        # Listen for messages from server
        async for message in websocket_connection:
            data = json.loads(message)
            message_queue.put(data)
            
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        websocket_connection = None

def start_websocket_client():
    """Start WebSocket client in a separate thread"""
    def run_client():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(connect_to_server())
    
    thread = threading.Thread(target=run_client, daemon=True)
    thread.start()

async def send_message(message):
    """Send message to server"""
    if websocket_connection:
        try:
            await websocket_connection.send(json.dumps(message))
        except Exception as e:
            print(f"Error sending message: {e}")

def send_message_sync(message):
    """Send message synchronously"""
    if websocket_connection:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(send_message(message))
        except Exception as e:
            pass  # Silently handle network errors to avoid disrupting drawing

def process_server_messages():
    """Process messages from server"""
    global client_id, user_name, user_color, connected_users, canvas
    
    while not message_queue.empty():
        try:
            data = message_queue.get_nowait()
            message_type = data.get('type')
            
            if message_type == 'welcome':
                client_id = data['client_id']
                user_name = data['user_name']
                user_color = data['color']
                print(f"Welcome! You are {user_name}")
                
            elif message_type == 'user_joined':
                connected_users[data['client_id']] = {
                    'name': data['user_name'],
                    'color': data['color']
                }
                print(f"{data['user_name']} joined the session")
                
            elif message_type == 'user_left':
                if data['client_id'] in connected_users:
                    print(f"{connected_users[data['client_id']]['name']} left the session")
                    del connected_users[data['client_id']]
                    
            elif message_type == 'drawing_event':
                handle_remote_drawing_event(data)
                
            elif message_type == 'drawing_batch':
                # Handle batched drawing events
                events = data.get('events', [])
                for event in events:
                    handle_remote_drawing_event(event)
                
            elif message_type == 'canvas_clear':
                canvas = np.full((WHITEBOARD_HEIGHT, WHITEBOARD_WIDTH, 3), 255, dtype=np.uint8)
                print(f"Canvas cleared by {data.get('user_name', 'Unknown')}")
                
            elif message_type == 'canvas_saved':
                print(f"Canvas saved by {data.get('user_name', 'Unknown')}")
                
            elif message_type == 'diagram_generated':
                handle_remote_diagram(data)
                
        except Exception as e:
            pass  # Silently handle processing errors

def handle_remote_drawing_event(data):
    """Handle drawing events from other users"""
    global canvas
    
    try:
        tool = data.get('tool', 'Freehand')
        color_bgr = tuple(data.get('color', [0, 0, 255]))
        thickness = data.get('thickness', 5)
        points = data.get('points', [])
        
        if tool == 'Freehand' and len(points) >= 2:
            for i in range(1, len(points)):
                pt1 = tuple(points[i-1])
                pt2 = tuple(points[i])
                if is_valid_point(pt1) and is_valid_point(pt2):
                    cv2.line(canvas, pt1, pt2, color_bgr, thickness)
                    cv2.circle(canvas, pt2, thickness//2, color_bgr, -1)
                
        elif tool == 'Line' and len(points) == 2:
            pt1, pt2 = tuple(points[0]), tuple(points[1])
            if is_valid_point(pt1) and is_valid_point(pt2):
                cv2.line(canvas, pt1, pt2, color_bgr, thickness)
            
        elif tool == 'Rectangle' and len(points) == 2:
            pt1, pt2 = tuple(points[0]), tuple(points[1])
            if is_valid_point(pt1) and is_valid_point(pt2):
                cv2.rectangle(canvas, pt1, pt2, color_bgr, thickness)
            
        elif tool == 'Circle' and len(points) == 2:
            center = tuple(points[0])
            if is_valid_point(center):
                radius = int(calculate_distance(points[0], points[1]))
                radius = max(1, min(radius, 200))  # Clamp radius
                cv2.circle(canvas, center, radius, color_bgr, thickness)
            
        elif tool == 'Eraser' and len(points) >= 1:
            for point in points:
                if is_valid_point(point):
                    cv2.circle(canvas, tuple(point), data.get('eraser_thickness', 30), (255, 255, 255), -1)
                
    except Exception as e:
        pass  # Silently handle drawing errors

def is_valid_point(point):
    """Check if point is within canvas bounds"""
    try:
        x, y = point
        return 0 <= x < WHITEBOARD_WIDTH and toolbar_height <= y < WHITEBOARD_HEIGHT
    except:
        return False

def handle_remote_diagram(data):
    """Handle diagram generation from other users"""
    print(f"Diagram generated by {data.get('user_name', 'Unknown')}")

def send_drawing_batch():
    """Send batched drawing events to reduce network traffic"""
    global drawing_batch, last_send_time
    
    current_time = time.time()
    if (len(drawing_batch) >= BATCH_SIZE or 
        (drawing_batch and current_time - last_send_time > SEND_INTERVAL)):
        
        if websocket_connection and drawing_batch:
            message = {
                'type': 'drawing_batch',
                'events': drawing_batch.copy()
            }
            
            # Send asynchronously to avoid blocking drawing
            def send_async():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(send_message(message))
                except:
                    pass
            
            threading.Thread(target=send_async, daemon=True).start()
            
        drawing_batch = []
        last_send_time = current_time

def add_to_drawing_batch(tool, color, thickness, points, **kwargs):
    """Add drawing event to batch for efficient network transmission"""
    global drawing_batch
    
    event = {
        'tool': tool,
        'color': color,
        'thickness': thickness,
        'points': points,
        **kwargs
    }
    
    drawing_batch.append(event)
    send_drawing_batch()

def send_canvas_clear():
    """Send canvas clear event"""
    if websocket_connection:
        message = {'type': 'canvas_clear'}
        threading.Thread(target=lambda: send_message_sync(message), daemon=True).start()

def send_canvas_save():
    """Send canvas save event"""
    if websocket_connection:
        # Encode canvas as base64
        _, buffer = cv2.imencode('.png', canvas)
        canvas_base64 = base64.b64encode(buffer).decode('utf-8')
        
        message = {
            'type': 'canvas_save',
            'canvas_data': canvas_base64
        }
        threading.Thread(target=lambda: send_message_sync(message), daemon=True).start()

def smooth_points(points, factor):
    """Apply enhanced smoothing to a list of points"""
    if len(points) < 2:
        return points[-1] if points else (0, 0)
    
    if len(points) >= 4:
        # Use weighted average for better smoothing
        recent_points = points[-4:]
        weights = [0.1, 0.2, 0.3, 0.4]  # More weight to recent points
        
        x_avg = sum(p[0] * w for p, w in zip(recent_points, weights))
        y_avg = sum(p[1] * w for p, w in zip(recent_points, weights))
        
        return (int(x_avg), int(y_avg))
    else:
        return points[-1]

def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=8):
    """Draw a rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def create_toolbar():
    """Create a professional-looking toolbar with collaboration info"""
    toolbar = np.full((toolbar_height, WHITEBOARD_WIDTH, 3), toolbar_color, dtype=np.uint8)
    
    # Add gradient background
    for y in range(toolbar_height):
        alpha = y / toolbar_height
        gradient_color = tuple(int(toolbar_color[i] + alpha * 10) for i in range(3))
        cv2.line(toolbar, (0, y), (WHITEBOARD_WIDTH, y), gradient_color, 1)
    
    # Add collaboration status
    if websocket_connection and user_name:
        status_text = f"Connected as {user_name} | Users online: {len(connected_users) + 1}"
        cv2.putText(toolbar, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    else:
        cv2.putText(toolbar, "Offline Mode - Enhanced Performance", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Rest of toolbar creation (same as original)
    section_margin = 25
    button_height = 50
    button_spacing = 12
    start_y = (toolbar_height - button_height) // 2 + 15
    current_x = section_margin
    
    # COLOR PALETTE SECTION
    cv2.putText(toolbar, "COLORS", (current_x, start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    color_size = 45
    color_spacing = 10
    color_start_y = start_y + 2
    
    for i, color in enumerate(colors):
        color_x = current_x + i * (color_size + color_spacing)
        color_end_x = color_x + color_size
        color_end_y = color_start_y + color_size
        
        cv2.rectangle(toolbar, (color_x + 2, color_start_y + 2), (color_end_x + 2, color_end_y + 2), shadow_color, -1)
        
        center = (color_x + color_size//2, color_start_y + color_size//2)
        cv2.circle(toolbar, center, color_size//2 - 2, color, -1)
        cv2.circle(toolbar, center, color_size//2 - 1, (200, 200, 200), 1)
        
        if i == current_color_index and current_tool_index != tools.index("Eraser"):
            cv2.circle(toolbar, center, color_size//2 + 3, highlight_color, 3)
    
    current_x += len(colors) * (color_size + color_spacing) + section_margin * 2
    
    # TOOLS SECTION
    cv2.putText(toolbar, "TOOLS", (current_x, start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    tool_width = 75
    tool_height = button_height
    
    for i, tool in enumerate(tools):
        tool_x = current_x + i * (tool_width + button_spacing)
        tool_end_x = tool_x + tool_width
        tool_end_y = start_y + tool_height
        
        draw_rounded_rect(toolbar, (tool_x + 2, start_y + 2), (tool_end_x + 2, tool_end_y + 2), shadow_color, -1, 6)
        
        button_bg_color = highlight_color if i == current_tool_index else button_color
        draw_rounded_rect(toolbar, (tool_x, start_y), (tool_end_x, tool_end_y), button_bg_color, -1, 6)
        
        text_size = cv2.getTextSize(tool, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        text_x = tool_x + (tool_width - text_size[0]) // 2
        text_y = start_y + (tool_height + text_size[1]) // 2
        
        text_col = (255, 255, 255) if i == current_tool_index else text_color
        cv2.putText(toolbar, tool, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1)
    
    current_x += len(tools) * (tool_width + button_spacing) + section_margin * 2
    
    # ACTION BUTTONS SECTION
    cv2.putText(toolbar, "ACTIONS", (current_x, start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    action_button_width = 80
    
    # Clear All button
    clear_x = current_x
    clear_end_x = clear_x + action_button_width
    clear_end_y = start_y + button_height
    
    draw_rounded_rect(toolbar, (clear_x + 2, start_y + 2), (clear_end_x + 2, clear_end_y + 2), shadow_color, -1, 6)
    draw_rounded_rect(toolbar, (clear_x, start_y), (clear_end_x, clear_end_y), clear_button_color, -1, 6)
    
    text_size = cv2.getTextSize("Clear All", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    text_x = clear_x + (action_button_width - text_size[0]) // 2
    text_y = start_y + (button_height + text_size[1]) // 2
    cv2.putText(toolbar, "Clear All", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Save button
    save_x = clear_end_x + button_spacing
    save_end_x = save_x + action_button_width
    save_end_y = start_y + button_height
    
    draw_rounded_rect(toolbar, (save_x + 2, start_y + 2), (save_end_x + 2, save_end_y + 2), shadow_color, -1, 6)
    draw_rounded_rect(toolbar, (save_x, start_y), (save_end_x, save_end_y), save_button_color, -1, 6)
    
    text_size = cv2.getTextSize("Save", cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    text_x = save_x + (action_button_width - text_size[0]) // 2
    text_y = start_y + (button_height + text_size[1]) // 2
    cv2.putText(toolbar, "Save", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Add separator lines
    separator_color = (80, 85, 95)
    sep1_x = section_margin + len(colors) * (color_size + color_spacing) + section_margin
    cv2.line(toolbar, (sep1_x, start_y), (sep1_x, start_y + button_height), separator_color, 2)
    
    sep2_x = sep1_x + section_margin + len(tools) * (tool_width + button_spacing) + section_margin
    cv2.line(toolbar, (sep2_x, start_y), (sep2_x, start_y + button_height), separator_color, 2)
    
    return toolbar

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def clamp_to_whiteboard(x, y):
    """Clamp coordinates to stay within whiteboard boundaries"""
    min_x = 0
    max_x = WHITEBOARD_WIDTH - 1
    min_y = toolbar_height
    max_y = WHITEBOARD_HEIGHT - 1
    
    clamped_x = max(min_x, min(max_x, x))
    clamped_y = max(min_y, min(max_y, y))
    
    return clamped_x, clamped_y

def clamp_circle_radius(center_x, center_y, radius):
    """Clamp circle radius to stay within whiteboard boundaries"""
    max_radius_x = min(center_x, WHITEBOARD_WIDTH - 1 - center_x)
    max_radius_y = min(center_y - toolbar_height, WHITEBOARD_HEIGHT - 1 - center_y)
    max_radius = min(max_radius_x, max_radius_y, radius)
    
    return max(1, max_radius)

def save_drawing():
    """Save the current drawing and process with Gemini API"""
    global canvas, save_button_clicked
    
    clean_canvas = canvas.copy()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"saved_drawings/drawing_{timestamp}.png"
    cv2.imwrite(filename, clean_canvas)
    print(f"Drawing saved as: {filename}")
    
    # Send save event to server
    send_canvas_save()
    
    if GEMINI_AVAILABLE:
        try:
            process_with_gemini(filename)
        except Exception as e:
            print(f"Error processing with Gemini: {e}")
    else:
        print("Gemini API not available. Only saved the image.")
    
    save_button_clicked = False

def process_with_gemini(image_path):
    """Process the saved image with Gemini API to convert sketch to diagram"""
    print("Processing image with Gemini API...")
    
    img = PIL.Image.open(image_path)
    
    prompt = """
    Analyze this hand-drawn sketch/diagram and convert it into a proper structured diagram description. 
    
    Please identify what type of diagram this appears to be (flowchart, UML diagram, system architecture, mind map, etc.) and provide:
    
    1. A clear description of what the diagram represents
    2. The components/elements identified in the sketch
    3. The relationships between components
    4. A properly formatted Mermaid diagram syntax
    5. Suggestions for improvement or completion if the diagram seems incomplete
    
    IMPORTANT: Always provide the diagram in Mermaid syntax format for best compatibility.
    Use this format:
    
    ```mermaid
    graph TD
        A[Start] --> B[Process]
        B --> C[Decision]
        C --> D[End]
    ```
    
    For flowcharts, use 'graph TD' (top-down) or 'graph LR' (left-right).
    Use square brackets [Text] for process boxes.
    Use arrows --> to connect elements.
    """
    
    try:
        response = model.generate_content([prompt, img])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_filename = f"saved_drawings/analysis_{timestamp}.txt"
        
        with open(analysis_filename, 'w') as f:
            f.write("=== GEMINI DIAGRAM ANALYSIS ===\n\n")
            f.write(response.text)
        
        print(f"Analysis saved as: {analysis_filename}")
        print("\n=== GEMINI ANALYSIS ===")
        print(response.text)
        print("=" * 50)
        
        # Extract and generate diagram automatically - MERMAID ONLY for now
        diagram_type, diagram_syntax = extract_diagram_syntax(response.text)
        
        if diagram_type and diagram_syntax:
            print(f"\n🎯 Found {diagram_type} syntax! Generating visual diagram...")
            
            # Only use Mermaid for now - more reliable
            if diagram_type == 'mermaid':
                diagram_image = generate_mermaid_diagram(diagram_syntax)
            elif diagram_type == 'plantuml':
                print("🔄 Converting PlantUML to Mermaid for better compatibility...")
                # Convert simple PlantUML to Mermaid
                mermaid_syntax = convert_plantuml_to_mermaid(diagram_syntax)
                diagram_image = generate_mermaid_diagram(mermaid_syntax)
                diagram_type = 'mermaid'  # Update type for display
            else:
                diagram_image = None
            
            if diagram_image:
                # Save the generated diagram
                diagram_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                diagram_filename = f"saved_drawings/generated_diagram_{diagram_timestamp}.png"
                
                with open(diagram_filename, 'wb') as f:
                    f.write(diagram_image)
                
                print(f"📊 Generated diagram saved as: {diagram_filename}")
                
                # Display the diagram automatically
                display_generated_diagram(diagram_image, diagram_type)
            else:
                print("❌ Failed to generate diagram from syntax")
        else:
            print("ℹ️  No diagram syntax found in analysis - try drawing a more detailed diagram!")
        
        # Send diagram to other users
        if websocket_connection:
            message = {
                'type': 'diagram_generated',
                'analysis': response.text,
                'filename': analysis_filename,
                'diagram_generated': diagram_type is not None
            }
            threading.Thread(target=lambda: send_message_sync(message), daemon=True).start()
        
    except Exception as e:
        print(f"Error generating content with Gemini: {e}")

def detect_toolbar_selection(x, y):
    """Detect if user is selecting an item in the professional toolbar"""
    global current_color_index, current_tool_index, canvas, save_button_clicked
    
    if y >= toolbar_height:
        return False
    
    section_margin = 25
    button_height = 50
    button_spacing = 12
    start_y = (toolbar_height - button_height) // 2 + 15
    current_x = section_margin
    
    # COLOR SELECTION
    color_size = 45
    color_spacing = 10
    color_start_y = start_y + 2
    
    for i in range(len(colors)):
        color_x = current_x + i * (color_size + color_spacing)
        color_end_x = color_x + color_size
        color_end_y = color_start_y + color_size
        
        if color_x <= x <= color_end_x and color_start_y <= y <= color_end_y:
            if current_tool_index == tools.index("Eraser"):
                current_tool_index = 0
            current_color_index = i
            return True
    
    current_x += len(colors) * (color_size + color_spacing) + section_margin * 2
    
    # TOOL SELECTION
    tool_width = 75
    tool_height = button_height
    
    for i in range(len(tools)):
        tool_x = current_x + i * (tool_width + button_spacing)
        tool_end_x = tool_x + tool_width
        tool_end_y = start_y + tool_height
        
        if tool_x <= x <= tool_end_x and start_y <= y <= tool_end_y:
            current_tool_index = i
            return True
    
    current_x += len(tools) * (tool_width + button_spacing) + section_margin * 2
    
    # ACTION BUTTONS
    action_button_width = 80
    
    # Clear All button
    clear_x = current_x
    clear_end_x = clear_x + action_button_width
    clear_end_y = start_y + button_height
    
    if clear_x <= x <= clear_end_x and start_y <= y <= clear_end_y:
        canvas = np.full((WHITEBOARD_HEIGHT, WHITEBOARD_WIDTH, 3), 255, dtype=np.uint8)
        send_canvas_clear()
        print("Canvas cleared!")
        return True
    
    # Save button
    save_x = clear_end_x + button_spacing
    save_end_x = save_x + action_button_width
    save_end_y = start_y + button_height
    
    if save_x <= x <= save_end_x and start_y <= y <= save_end_y:
        save_button_clicked = True
        return True
    
    return False

def get_gesture_state(hand_landmarks):
    """Detect drawing gesture state based on hand landmarks"""
    if not hand_landmarks:
        return False, None
    
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    middle_tip = hand_landmarks.landmark[12]
    
    index_tip_y = int(index_tip.y * WHITEBOARD_HEIGHT)
    index_pip_y = int(index_pip.y * WHITEBOARD_HEIGHT)
    middle_tip_y = int(middle_tip.y * WHITEBOARD_HEIGHT)
    
    drawing_gesture = (index_tip_y < index_pip_y) and (middle_tip_y > index_pip_y)
    
    return drawing_gesture, (int(index_tip.x * WHITEBOARD_WIDTH), int(index_tip.y * WHITEBOARD_HEIGHT))

def generate_mermaid_diagram(mermaid_code):
    """Generate diagram from Mermaid syntax using mermaid.ink API"""
    try:
        # Clean and fix the mermaid code
        mermaid_code = mermaid_code.strip()
        
        # Fix common syntax issues
        if not mermaid_code.startswith('graph') and not mermaid_code.startswith('flowchart'):
            # If it doesn't start with graph/flowchart, add it
            mermaid_code = f"graph TD\n{mermaid_code}"
        
        # Replace problematic characters and fix syntax
        mermaid_code = mermaid_code.replace('{', '(').replace('}', ')')  # Fix decision nodes
        
        print(f"Cleaned Mermaid code:\n{mermaid_code}")
        
        # Try multiple encoding methods
        methods = [
            lambda code: base64.b64encode(code.encode('utf-8')).decode('utf-8'),
            lambda code: urllib.parse.quote(code, safe=''),
            lambda code: base64.urlsafe_b64encode(code.encode('utf-8')).decode('utf-8')
        ]
        
        for i, encode_method in enumerate(methods):
            try:
                encoded_code = encode_method(mermaid_code)
                url = f"https://mermaid.ink/img/{encoded_code}"
                
                print(f"Trying method {i+1}: {url[:100]}...")
                response = requests.get(url, timeout=15)
                
                if response.status_code == 200:
                    print(f"✅ Success with method {i+1}!")
                    return response.content
                else:
                    print(f"Method {i+1} failed: {response.status_code}")
                    
            except Exception as method_error:
                print(f"Method {i+1} error: {method_error}")
                continue
        
        # If all methods fail, try a simple fallback diagram
        print("All methods failed, creating fallback diagram...")
        fallback_code = """graph TD
    A[Start] --> B[Process]
    B --> C[End]"""
        
        encoded_fallback = base64.b64encode(fallback_code.encode('utf-8')).decode('utf-8')
        fallback_url = f"https://mermaid.ink/img/{encoded_fallback}"
        
        response = requests.get(fallback_url, timeout=10)
        if response.status_code == 200:
            print("✅ Fallback diagram generated!")
            return response.content
        
        return None
        
    except Exception as e:
        print(f"Error generating Mermaid diagram: {e}")
        return None

def generate_plantuml_diagram(plantuml_code):
    """Generate diagram from PlantUML syntax using PlantUML server"""
    try:
        # Clean the PlantUML code
        plantuml_code = plantuml_code.strip()
        if not plantuml_code.startswith('@startuml'):
            plantuml_code = '@startuml\n' + plantuml_code + '\n@enduml'
        
        print(f"Cleaned PlantUML code:\n{plantuml_code}")
        
        # Try multiple PlantUML servers and encoding methods
        servers = [
            "http://www.plantuml.com/plantuml",
            "https://plantuml-server.kkeisuke.dev", 
            "http://plantuml.com:8080/plantuml"
        ]
        
        for server_url in servers:
            try:
                print(f"Trying PlantUML server: {server_url}")
                
                # Method 1: Simple base64 encoding
                try:
                    encoded = base64.b64encode(plantuml_code.encode('utf-8')).decode('utf-8')
                    url = f"{server_url}/png/{encoded}"
                    
                    response = requests.get(url, timeout=15)
                    if response.status_code == 200:
                        print("✅ Success with base64 encoding!")
                        return response.content
                    else:
                        print(f"Base64 method failed: {response.status_code}")
                except Exception as e:
                    print(f"Base64 method error: {e}")
                
                # Method 2: URL encoding
                try:
                    encoded = urllib.parse.quote(plantuml_code, safe='')
                    url = f"{server_url}/png/{encoded}"
                    
                    response = requests.get(url, timeout=15)
                    if response.status_code == 200:
                        print("✅ Success with URL encoding!")
                        return response.content
                    else:
                        print(f"URL encoding method failed: {response.status_code}")
                except Exception as e:
                    print(f"URL encoding method error: {e}")
                
                # Method 3: POST request with form data
                try:
                    data = {'text': plantuml_code}
                    response = requests.post(f"{server_url}/png", data=data, timeout=15)
                    if response.status_code == 200:
                        print("✅ Success with POST method!")
                        return response.content
                    else:
                        print(f"POST method failed: {response.status_code}")
                except Exception as e:
                    print(f"POST method error: {e}")
                    
            except Exception as server_error:
                print(f"Server {server_url} failed: {server_error}")
                continue
        
        # If all methods fail, create a simple fallback PlantUML diagram
        print("All PlantUML methods failed, creating fallback diagram...")
        fallback_code = """@startuml
start
:Process;
stop
@enduml"""
        
        try:
            encoded_fallback = base64.b64encode(fallback_code.encode('utf-8')).decode('utf-8')
            fallback_url = f"http://www.plantuml.com/plantuml/png/{encoded_fallback}"
            
            response = requests.get(fallback_url, timeout=10)
            if response.status_code == 200:
                print("✅ Fallback PlantUML diagram generated!")
                return response.content
        except Exception as e:
            print(f"Fallback failed: {e}")
        
        return None
        
    except Exception as e:
        print(f"Error generating PlantUML diagram: {e}")
        return None

def convert_plantuml_to_mermaid(plantuml_code):
    """Convert simple PlantUML syntax to Mermaid for better compatibility"""
    try:
        # Simple conversion for basic PlantUML activity diagrams
        mermaid_lines = ["graph TD"]
        
        # Remove PlantUML wrapper
        plantuml_code = plantuml_code.replace('@startuml', '').replace('@enduml', '').strip()
        
        lines = plantuml_code.split('\n')
        node_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line == 'start':
                mermaid_lines.append(f"    A{node_counter}[Start]")
                node_counter += 1
            elif line == 'stop' or line == 'end':
                mermaid_lines.append(f"    A{node_counter}[End]")
                node_counter += 1
            elif line.startswith(':') and line.endswith(';'):
                # Activity
                activity = line[1:-1]  # Remove : and ;
                mermaid_lines.append(f"    A{node_counter}[{activity}]")
                if node_counter > 1:
                    mermaid_lines.append(f"    A{node_counter-1} --> A{node_counter}")
                node_counter += 1
        
        # If we have nodes, connect them
        if node_counter > 2:
            # Connect the last two nodes if not already connected
            if "-->" not in mermaid_lines[-1]:
                mermaid_lines.append(f"    A{node_counter-2} --> A{node_counter-1}")
        
        result = '\n'.join(mermaid_lines)
        print(f"Converted PlantUML to Mermaid:\n{result}")
        return result
        
    except Exception as e:
        print(f"Error converting PlantUML to Mermaid: {e}")
        # Return a simple fallback
        return """graph TD
    A[Start] --> B[Process]
    B --> C[End]"""

def extract_diagram_syntax(analysis_text):
    """Extract Mermaid or PlantUML syntax from Gemini analysis"""
    mermaid_patterns = [
        r'```mermaid\n(.*?)\n```',
        r'```\n(graph.*?)\n```',
        r'```\n(flowchart.*?)\n```',
        r'(graph\s+TD.*?)(?=\n\n|\n[A-Z]|\Z)',
        r'(flowchart\s+TD.*?)(?=\n\n|\n[A-Z]|\Z)',
        r'(graph\s+\w+.*?)(?=\n\n|\n[A-Z]|\Z)',
        r'(flowchart\s+\w+.*?)(?=\n\n|\n[A-Z]|\Z)',
        # More specific patterns for the format we saw
        r'```mermaid\s*\n(graph\s+TD.*?)\n```',
        r'Mermaid.*?:\s*```mermaid\s*\n(.*?)\n```'
    ]
    
    plantuml_patterns = [
        r'```plantuml\n(.*?)\n```',
        r'```\n(@startuml.*?@enduml)\n```',
        r'(@startuml.*?@enduml)',
        r'```uml\n(.*?)\n```',
        # More specific PlantUML patterns
        r'PlantUML.*?:\s*```plantuml\s*\n(.*?)\n```',
        r'PlantUML.*?:\s*```\s*\n(@startuml.*?@enduml)\n```',
        r'PlantUML.*?:\s*(@startuml.*?@enduml)',
        # Handle cases where PlantUML syntax is mentioned without code blocks
        r'PlantUML.*?syntax.*?:\s*\n\n(@startuml.*?@enduml)',
        r'UML.*?diagram.*?:\s*\n\n(@startuml.*?@enduml)'
    ]
    
    print("🔍 Searching for diagram syntax in analysis...")
    
    # Try to find Mermaid syntax
    for i, pattern in enumerate(mermaid_patterns):
        match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
        if match:
            syntax = match.group(1).strip()
            print(f"✅ Found Mermaid syntax with pattern {i+1}: {syntax[:50]}...")
            return 'mermaid', syntax
    
    # Try to find PlantUML syntax
    for i, pattern in enumerate(plantuml_patterns):
        match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
        if match:
            syntax = match.group(1).strip()
            print(f"✅ Found PlantUML syntax with pattern {i+1}: {syntax[:50]}...")
            return 'plantuml', syntax
    
    print("❌ No diagram syntax found in analysis")
    return None, None

def display_generated_diagram(image_data, diagram_type):
    """Display the generated diagram in a new window"""
    try:
        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        diagram_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if diagram_img is not None:
            # Resize if too large
            height, width = diagram_img.shape[:2]
            max_size = 800
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                diagram_img = cv2.resize(diagram_img, (new_width, new_height))
            
            # Display the diagram
            cv2.imshow(f"Generated {diagram_type.title()} Diagram", diagram_img)
            print(f"✨ {diagram_type.title()} diagram displayed! Press any key in diagram window to close.")
            
            return True
        else:
            print("Failed to decode generated diagram")
            return False
    except Exception as e:
        print(f"Error displaying diagram: {e}")
        return False

def main():
    global drawing_mode, prev_points, canvas, start_x, start_y, preview_shape, gesture_start_time, shape_started, save_button_clicked
    
    print("Optimized Collaborative Virtual Painter - Controls:")
    print("- Point with index finger to draw")
    print("- Keep middle finger down while drawing")
    print("- Select colors and tools from toolbar")
    print("- Enhanced performance with smooth drawing!")
    print("- Press 'q' to quit")
    
    # Start WebSocket client (but don't block if it fails)
    try:
        start_websocket_client()
        time.sleep(1)  # Reduced connection wait time
    except:
        print("Running in offline mode for best performance")
    
    # Performance optimization: pre-compile some operations
    frame_count = 0
    fps_start_time = time.time()
    
    while True:
        frame_count += 1
        
        # Process server messages (but don't block)
        try:
            process_server_messages()
        except:
            pass
        
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        whiteboard_display = canvas.copy()
        toolbar = create_toolbar()
        whiteboard_display[0:toolbar_height, 0:WHITEBOARD_WIDTH] = toolbar
        
        camera_preview = cv2.resize(img, (CAMERA_WIDTH, CAMERA_HEIGHT))
        combined_display = np.full((WHITEBOARD_HEIGHT, TOTAL_WIDTH, 3), 240, dtype=np.uint8)
        combined_display[0:WHITEBOARD_HEIGHT, 0:WHITEBOARD_WIDTH] = whiteboard_display
        combined_display[CAMERA_Y_OFFSET:CAMERA_Y_OFFSET+CAMERA_HEIGHT, WHITEBOARD_WIDTH:TOTAL_WIDTH] = camera_preview
        
        if save_button_clicked:
            save_drawing()
        
        gesture_detected = False
        current_point = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on camera preview
                scaled_landmarks = []
                for lm in hand_landmarks.landmark:
                    scaled_x = int(lm.x * CAMERA_WIDTH)
                    scaled_y = int(lm.y * CAMERA_HEIGHT)
                    scaled_landmarks.append([scaled_x, scaled_y])
                
                for i, (x, y) in enumerate(scaled_landmarks):
                    cv2.circle(camera_preview, (x, y), 3, (0, 255, 0), -1)
                
                gesture_detected, current_point = get_gesture_state(hand_landmarks)
                
                index_tip = hand_landmarks.landmark[8]
                finger_point = (int(index_tip.x * WHITEBOARD_WIDTH), int(index_tip.y * WHITEBOARD_HEIGHT))
                
                # Enhanced pointer visualization
                if drawing_mode:
                    cv2.circle(whiteboard_display, finger_point, 15, (0, 255, 0), -1)
                    cv2.circle(whiteboard_display, finger_point, 18, (0, 255, 0), 2)
                elif gesture_detected:
                    cv2.circle(whiteboard_display, finger_point, 15, (0, 255, 255), -1)
                    cv2.circle(whiteboard_display, finger_point, 18, (0, 255, 255), 2)
                else:
                    cv2.circle(whiteboard_display, finger_point, 12, (0, 0, 255), -1)
                    cv2.circle(whiteboard_display, finger_point, 15, (0, 0, 255), 2)
                
                if gesture_detected and current_point:
                    if detect_toolbar_selection(current_point[0], current_point[1]):
                        gesture_start_time = 0
                        drawing_mode = False
                        shape_started = False
                        prev_points = []
                        continue
                    
                    if not drawing_mode:
                        if gesture_start_time == 0:
                            gesture_start_time = time.time()
                        elif time.time() - gesture_start_time >= GESTURE_DELAY:
                            drawing_mode = True
                            if current_tool_index in [1, 2, 3] and not shape_started:
                                start_x, start_y = clamp_to_whiteboard(current_point[0], current_point[1])
                                shape_started = True
                            prev_points = [current_point]
                    
                    if drawing_mode and current_point[1] > toolbar_height:
                        clamped_point = clamp_to_whiteboard(current_point[0], current_point[1])
                        
                        if current_tool_index == tools.index("Eraser"):
                            cv2.circle(canvas, clamped_point, eraser_thickness, (255, 255, 255), -1)
                            # Add to batch for network sync
                            add_to_drawing_batch("Eraser", [255, 255, 255], eraser_thickness, [clamped_point], eraser_thickness=eraser_thickness)
                            
                        elif current_tool_index == 0:  # Freehand
                            prev_points.append(clamped_point)
                            if len(prev_points) > MAX_POINTS:
                                prev_points.pop(0)
                            
                            if len(prev_points) >= 2:
                                # Enhanced smooth drawing
                                smoothed_point = smooth_points(prev_points, smoothing_factor)
                                
                                # Draw multiple connecting lines for ultra-smooth appearance
                                for i in range(1, len(prev_points)):
                                    pt1 = prev_points[i-1]
                                    pt2 = prev_points[i]
                                    # Draw thick line with tapered ends
                                    cv2.line(canvas, pt1, pt2, colors[current_color_index], brush_thickness)
                                    cv2.circle(canvas, pt2, brush_thickness//2, colors[current_color_index], -1)
                                
                                # Add to batch every few points to reduce network traffic
                                if len(prev_points) % 3 == 0:  # Send every 3rd point
                                    add_to_drawing_batch("Freehand", list(colors[current_color_index]), brush_thickness, prev_points[-3:])
                        
                        elif current_tool_index in [1, 2, 3] and shape_started:
                            preview_shape = canvas.copy()
                            clamped_start = clamp_to_whiteboard(start_x, start_y)
                            
                            if current_tool_index == 1:  # Line
                                cv2.line(preview_shape, clamped_start, clamped_point, colors[current_color_index], brush_thickness)
                            elif current_tool_index == 2:  # Rectangle
                                top_left = (min(clamped_start[0], clamped_point[0]), min(clamped_start[1], clamped_point[1]))
                                bottom_right = (max(clamped_start[0], clamped_point[0]), max(clamped_start[1], clamped_point[1]))
                                cv2.rectangle(preview_shape, top_left, bottom_right, colors[current_color_index], brush_thickness)
                            else:  # Circle
                                radius = int(calculate_distance(clamped_start, clamped_point))
                                clamped_radius = clamp_circle_radius(clamped_start[0], clamped_start[1], radius)
                                cv2.circle(preview_shape, clamped_start, clamped_radius, colors[current_color_index], brush_thickness)
                else:
                    if drawing_mode:
                        if current_tool_index in [1, 2, 3] and preview_shape is not None and shape_started:
                            canvas = preview_shape.copy()
                            
                            # Send completed shape to other users
                            tool_name = tools[current_tool_index]
                            clamped_start = clamp_to_whiteboard(start_x, start_y)
                            last_point = clamp_to_whiteboard(prev_points[-1][0], prev_points[-1][1]) if prev_points else clamped_start
                            add_to_drawing_batch(tool_name, list(colors[current_color_index]), brush_thickness, [clamped_start, last_point])
                            
                            preview_shape = None
                            shape_started = False
                        drawing_mode = False
                        gesture_start_time = 0
                        prev_points = []
        
        # Always send any remaining batched data
        send_drawing_batch()
        
        display_canvas = preview_shape if preview_shape is not None else canvas
        combined_display[0:WHITEBOARD_HEIGHT, 0:WHITEBOARD_WIDTH] = whiteboard_display
        
        if preview_shape is not None:
            combined_display[toolbar_height:WHITEBOARD_HEIGHT, 0:WHITEBOARD_WIDTH] = preview_shape[toolbar_height:WHITEBOARD_HEIGHT, 0:WHITEBOARD_WIDTH]
        
        combined_display[CAMERA_Y_OFFSET:CAMERA_Y_OFFSET+CAMERA_HEIGHT, WHITEBOARD_WIDTH:TOTAL_WIDTH] = camera_preview
        
        # Show FPS for performance monitoring
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            cv2.putText(combined_display, f"FPS: {fps:.1f}", (WHITEBOARD_WIDTH - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            fps_start_time = time.time()
        
        cv2.imshow("Optimized Collaborative Virtual Whiteboard", combined_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
