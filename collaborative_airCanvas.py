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
smoothing_factor = 0.5
save_button_clicked = False

# Points for shape drawing
start_x, start_y = 0, 0
preview_shape = None
shape_started = False

# Canvas to draw on (white whiteboard)
canvas = np.full((WHITEBOARD_HEIGHT, WHITEBOARD_WIDTH, 3), 255, dtype=np.uint8)

# Previous positions for smooth drawing
prev_points = []
MAX_POINTS = 5

# Gesture states
gesture_start_time = 0
GESTURE_DELAY = 0.1

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
            print(f"Error sending message sync: {e}")

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
                
            elif message_type == 'canvas_clear':
                canvas = np.full((WHITEBOARD_HEIGHT, WHITEBOARD_WIDTH, 3), 255, dtype=np.uint8)
                print(f"Canvas cleared by {data.get('user_name', 'Unknown')}")
                
            elif message_type == 'canvas_saved':
                print(f"Canvas saved by {data.get('user_name', 'Unknown')}")
                
            elif message_type == 'diagram_generated':
                handle_remote_diagram(data)
                
            elif message_type == 'user_cursor':
                # Update other users' cursor positions
                pass
                
        except Exception as e:
            print(f"Error processing server message: {e}")

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
                cv2.line(canvas, pt1, pt2, color_bgr, thickness)
                cv2.circle(canvas, pt2, thickness//2, color_bgr, -1)
                
        elif tool == 'Line' and len(points) == 2:
            cv2.line(canvas, tuple(points[0]), tuple(points[1]), color_bgr, thickness)
            
        elif tool == 'Rectangle' and len(points) == 2:
            cv2.rectangle(canvas, tuple(points[0]), tuple(points[1]), color_bgr, thickness)
            
        elif tool == 'Circle' and len(points) == 2:
            center = tuple(points[0])
            radius = int(calculate_distance(points[0], points[1]))
            cv2.circle(canvas, center, radius, color_bgr, thickness)
            
        elif tool == 'Eraser' and len(points) >= 1:
            for point in points:
                cv2.circle(canvas, tuple(point), data.get('eraser_thickness', 30), (255, 255, 255), -1)
                
    except Exception as e:
        print(f"Error handling remote drawing event: {e}")

def handle_remote_diagram(data):
    """Handle diagram generation from other users"""
    print(f"Diagram generated by {data.get('user_name', 'Unknown')}")
    # You can add logic here to display shared diagrams

def send_drawing_event(tool, color, thickness, points, **kwargs):
    """Send drawing event to server"""
    if websocket_connection:
        message = {
            'type': 'drawing_event',
            'tool': tool,
            'color': color,
            'thickness': thickness,
            'points': points,
            **kwargs
        }
        
        # Send in a separate thread to avoid blocking
        def send_async():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(send_message(message))
            except:
                pass
        
        threading.Thread(target=send_async, daemon=True).start()

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

# Keep all the existing utility functions from the original code
def smooth_points(points, factor):
    """Apply smoothing to a list of points"""
    if len(points) < 2:
        return points[-1] if points else (0, 0)
    
    if len(points) >= 3:
        recent_points = points[-3:]
        x_avg = sum(p[0] for p in recent_points) // len(recent_points)
        y_avg = sum(p[1] for p in recent_points) // len(recent_points)
        return (x_avg, y_avg)
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
        cv2.putText(toolbar, "Offline Mode", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Rest of toolbar creation (same as original)
    section_margin = 25
    button_height = 50
    button_spacing = 12
    start_y = (toolbar_height - button_height) // 2 + 15  # Adjusted for status text
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
    4. A properly formatted text representation (like Mermaid syntax, PlantUML, or structured text)
    5. Suggestions for improvement or completion if the diagram seems incomplete
    
    If this appears to be a UML diagram, provide PlantUML syntax.
    If this appears to be a flowchart, provide Mermaid flowchart syntax.
    For other diagram types, provide appropriate structured representation.
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
        
        # Send diagram to other users
        if websocket_connection:
            message = {
                'type': 'diagram_generated',
                'analysis': response.text,
                'filename': analysis_filename
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
    start_y = (toolbar_height - button_height) // 2 + 15  # Adjusted for status text
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
        send_canvas_clear()  # Notify other users
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

def main():
    global drawing_mode, prev_points, canvas, start_x, start_y, preview_shape, gesture_start_time, shape_started, save_button_clicked
    
    print("Collaborative Virtual Painter - Controls:")
    print("- Point with index finger to draw")
    print("- Keep middle finger down while drawing")
    print("- Select colors and tools from toolbar")
    print("- All actions are synchronized with other users!")
    print("- Press 'q' to quit")
    
    # Start WebSocket client
    start_websocket_client()
    time.sleep(2)  # Give time to connect
    
    while True:
        # Process server messages
        process_server_messages()
        
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
                
                if drawing_mode:
                    cv2.circle(whiteboard_display, finger_point, 12, (0, 255, 0), -1)
                    cv2.circle(whiteboard_display, finger_point, 15, (0, 255, 0), 2)
                elif gesture_detected:
                    cv2.circle(whiteboard_display, finger_point, 12, (0, 255, 255), -1)
                    cv2.circle(whiteboard_display, finger_point, 15, (0, 255, 255), 2)
                else:
                    cv2.circle(whiteboard_display, finger_point, 10, (0, 0, 255), -1)
                    cv2.circle(whiteboard_display, finger_point, 13, (0, 0, 255), 2)
                
                if gesture_detected:
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
                            # Send eraser event to other users
                            send_drawing_event("Eraser", [255, 255, 255], eraser_thickness, [clamped_point], eraser_thickness=eraser_thickness)
                            
                        elif current_tool_index == 0:  # Freehand
                            prev_points.append(clamped_point)
                            if len(prev_points) > MAX_POINTS:
                                prev_points.pop(0)
                            
                            if len(prev_points) >= 2:
                                for i in range(1, len(prev_points)):
                                    cv2.line(canvas, prev_points[i-1], prev_points[i], colors[current_color_index], brush_thickness)
                                cv2.circle(canvas, clamped_point, brush_thickness//2, colors[current_color_index], -1)
                                
                                # Send drawing event to other users
                                send_drawing_event("Freehand", list(colors[current_color_index]), brush_thickness, prev_points)
                        
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
                            
                            # Send shape event to other users
                            tool_name = tools[current_tool_index]
                            clamped_start = clamp_to_whiteboard(start_x, start_y)
                            if current_tool_index == 3:  # Circle
                                last_point = clamp_to_whiteboard(prev_points[-1][0], prev_points[-1][1]) if prev_points else clamped_start
                                send_drawing_event(tool_name, list(colors[current_color_index]), brush_thickness, [clamped_start, last_point])
                            else:
                                last_point = clamp_to_whiteboard(prev_points[-1][0], prev_points[-1][1]) if prev_points else clamped_start
                                send_drawing_event(tool_name, list(colors[current_color_index]), brush_thickness, [clamped_start, last_point])
                            
                            preview_shape = None
                            shape_started = False
                        drawing_mode = False
                        gesture_start_time = 0
                        prev_points = []
        
        display_canvas = preview_shape if preview_shape is not None else canvas
        combined_display[0:WHITEBOARD_HEIGHT, 0:WHITEBOARD_WIDTH] = whiteboard_display
        
        if preview_shape is not None:
            combined_display[toolbar_height:WHITEBOARD_HEIGHT, 0:WHITEBOARD_WIDTH] = preview_shape[toolbar_height:WHITEBOARD_HEIGHT, 0:WHITEBOARD_WIDTH]
        
        combined_display[CAMERA_Y_OFFSET:CAMERA_Y_OFFSET+CAMERA_HEIGHT, WHITEBOARD_WIDTH:TOTAL_WIDTH] = camera_preview
        
        cv2.imshow("Collaborative Virtual Whiteboard", combined_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
