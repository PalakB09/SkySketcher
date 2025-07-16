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

# Configure Gemini API
GEMINI_API_KEY = 'AIzaSyA_AOK7n5yHN-hQDRBqmeYaRmOhJp5xXwM'
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    GEMINI_AVAILABLE = True
    print("Gemini API configured successfully!")
except Exception as e:
    print(f"Gemini API not configured: {e}")
    GEMINI_AVAILABLE = False

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Layout settings
WHITEBOARD_WIDTH = 1200  # Increased to fit all buttons
WHITEBOARD_HEIGHT = 720
CAMERA_WIDTH = 280
CAMERA_HEIGHT = 210
CAMERA_Y_OFFSET = 100  # Push camera down from top
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
toolbar_color = (45, 45, 50)  # Modern dark background
button_color = (70, 75, 85)   # Professional gray for buttons
button_hover_color = (90, 95, 105)  # Lighter gray for hover
text_color = (240, 240, 245)  # Off-white text
highlight_color = (52, 152, 219)  # Professional blue highlight
save_button_color = (46, 204, 113)  # Modern green for save
clear_button_color = (231, 76, 60)  # Modern red for clear
shadow_color = (20, 20, 25)  # Dark shadow

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
smoothing_factor = 0.5  # For drawing smoothing (0 to 1)
save_button_clicked = False

# Points for shape drawing
start_x, start_y = 0, 0
preview_shape = None
shape_started = False

# Canvas to draw on (white whiteboard)
canvas = np.full((WHITEBOARD_HEIGHT, WHITEBOARD_WIDTH, 3), 255, dtype=np.uint8)  # White canvas

# Previous positions for smooth drawing
prev_points = []
MAX_POINTS = 5  # Number of points to use for smoothing

# Gesture states
gesture_start_time = 0
GESTURE_DELAY = 0.1  # Seconds to hold gesture before activating (reduced for better responsiveness)

# Create saves directory
if not os.path.exists('saved_drawings'):
    os.makedirs('saved_drawings')

def smooth_points(points, factor):
    """Apply smoothing to a list of points"""
    if len(points) < 2:
        return points[-1] if points else (0, 0)
    
    # Simple moving average for better smoothness
    if len(points) >= 3:
        # Use the last 3 points for smoothing
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
    
    # Draw main rectangle
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def create_toolbar():
    """Create a professional-looking toolbar"""
    toolbar = np.full((toolbar_height, WHITEBOARD_WIDTH, 3), toolbar_color, dtype=np.uint8)
    
    # Add subtle gradient background
    for y in range(toolbar_height):
        alpha = y / toolbar_height
        gradient_color = tuple(int(toolbar_color[i] + alpha * 10) for i in range(3))
        cv2.line(toolbar, (0, y), (WHITEBOARD_WIDTH, y), gradient_color, 1)
    
    # Section spacing and sizing
    section_margin = 25
    button_height = 50
    button_spacing = 12
    start_y = (toolbar_height - button_height) // 2
    current_x = section_margin
    
    # === COLOR PALETTE SECTION ===
    # Section title
    cv2.putText(toolbar, "COLORS", (current_x, start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    # Color selection boxes with professional styling
    color_size = 45  # Increased from 35 for better visibility
    color_spacing = 10  # Slightly increased spacing too
    color_start_y = start_y + 2  # Adjusted to fit better
    
    for i, color in enumerate(colors):
        color_x = current_x + i * (color_size + color_spacing)
        color_end_x = color_x + color_size
        color_end_y = color_start_y + color_size
        
        # Draw shadow
        cv2.rectangle(toolbar, (color_x + 2, color_start_y + 2), (color_end_x + 2, color_end_y + 2), shadow_color, -1)
        
        # Draw color circle
        center = (color_x + color_size//2, color_start_y + color_size//2)
        cv2.circle(toolbar, center, color_size//2 - 2, color, -1)
        cv2.circle(toolbar, center, color_size//2 - 1, (200, 200, 200), 1)
        
        # Highlight selected color
        if i == current_color_index and current_tool_index != tools.index("Eraser"):
            cv2.circle(toolbar, center, color_size//2 + 3, highlight_color, 3)
    
    current_x += len(colors) * (color_size + color_spacing) + section_margin * 2
    
    # === TOOLS SECTION ===
    cv2.putText(toolbar, "TOOLS", (current_x, start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    tool_width = 75
    tool_height = button_height
    
    for i, tool in enumerate(tools):
        tool_x = current_x + i * (tool_width + button_spacing)
        tool_end_x = tool_x + tool_width
        tool_end_y = start_y + tool_height
        
        # Draw shadow
        draw_rounded_rect(toolbar, (tool_x + 2, start_y + 2), (tool_end_x + 2, tool_end_y + 2), shadow_color, -1, 6)
        
        # Draw button background
        button_bg_color = highlight_color if i == current_tool_index else button_color
        draw_rounded_rect(toolbar, (tool_x, start_y), (tool_end_x, tool_end_y), button_bg_color, -1, 6)
        
        # Add tool name
        text_size = cv2.getTextSize(tool, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        text_x = tool_x + (tool_width - text_size[0]) // 2
        text_y = start_y + (tool_height + text_size[1]) // 2
        
        text_col = (255, 255, 255) if i == current_tool_index else text_color
        cv2.putText(toolbar, tool, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1)
    
    current_x += len(tools) * (tool_width + button_spacing) + section_margin * 2
    
    # === ACTION BUTTONS SECTION ===
    cv2.putText(toolbar, "ACTIONS", (current_x, start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    action_button_width = 80
    
    # Clear All button (comes first now)
    clear_x = current_x
    clear_end_x = clear_x + action_button_width
    clear_end_y = start_y + button_height
    
    # Draw shadow
    draw_rounded_rect(toolbar, (clear_x + 2, start_y + 2), (clear_end_x + 2, clear_end_y + 2), shadow_color, -1, 6)
    
    # Draw Clear All button
    draw_rounded_rect(toolbar, (clear_x, start_y), (clear_end_x, clear_end_y), clear_button_color, -1, 6)
    
    # Add "Clear All" text
    text_size = cv2.getTextSize("Clear All", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    text_x = clear_x + (action_button_width - text_size[0]) // 2
    text_y = start_y + (button_height + text_size[1]) // 2
    cv2.putText(toolbar, "Clear All", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Save button (comes second now)
    save_x = clear_end_x + button_spacing
    save_end_x = save_x + action_button_width
    save_end_y = start_y + button_height
    
    # Draw shadow
    draw_rounded_rect(toolbar, (save_x + 2, start_y + 2), (save_end_x + 2, save_end_y + 2), shadow_color, -1, 6)
    
    # Draw Save button
    draw_rounded_rect(toolbar, (save_x, start_y), (save_end_x, save_end_y), save_button_color, -1, 6)
    
    # Add "Save" text
    text_size = cv2.getTextSize("Save", cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    text_x = save_x + (action_button_width - text_size[0]) // 2
    text_y = start_y + (button_height + text_size[1]) // 2
    cv2.putText(toolbar, "Save", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Add separator lines
    separator_color = (80, 85, 95)
    # Between colors and tools
    sep1_x = section_margin + len(colors) * (color_size + color_spacing) + section_margin
    cv2.line(toolbar, (sep1_x, start_y), (sep1_x, start_y + button_height), separator_color, 2)
    
    # Between tools and actions
    sep2_x = sep1_x + section_margin + len(tools) * (tool_width + button_spacing) + section_margin
    cv2.line(toolbar, (sep2_x, start_y), (sep2_x, start_y + button_height), separator_color, 2)
    
    return toolbar

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def clamp_to_whiteboard(x, y):
    """Clamp coordinates to stay within whiteboard boundaries"""
    # Account for toolbar height at the top
    min_x = 0
    max_x = WHITEBOARD_WIDTH - 1
    min_y = toolbar_height  # Don't draw on toolbar
    max_y = WHITEBOARD_HEIGHT - 1
    
    clamped_x = max(min_x, min(max_x, x))
    clamped_y = max(min_y, min(max_y, y))
    
    return clamped_x, clamped_y

def clamp_circle_radius(center_x, center_y, radius):
    """Clamp circle radius to stay within whiteboard boundaries"""
    # Calculate maximum radius that keeps circle within bounds
    max_radius_x = min(center_x, WHITEBOARD_WIDTH - 1 - center_x)
    max_radius_y = min(center_y - toolbar_height, WHITEBOARD_HEIGHT - 1 - center_y)
    max_radius = min(max_radius_x, max_radius_y, radius)
    
    return max(1, max_radius)  # Minimum radius of 1

def save_drawing():
    """Save the current drawing and process with Gemini API"""
    global canvas, save_button_clicked
    
    # Create a clean canvas (without camera overlay)
    clean_canvas = canvas.copy()
    
    # Save the image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"saved_drawings/drawing_{timestamp}.png"
    cv2.imwrite(filename, clean_canvas)
    print(f"Drawing saved as: {filename}")
    
    # Process with Gemini API if available
    if GEMINI_AVAILABLE:
        try:
            process_with_gemini(filename)
        except Exception as e:
            print(f"Error processing with Gemini: {e}")
    else:
        print("Gemini API not available. Only saved the image.")
    
    save_button_clicked = False

def generate_mermaid_diagram(mermaid_code):
    """Generate diagram from Mermaid syntax using mermaid.ink API"""
    try:
        # Clean the mermaid code
        mermaid_code = mermaid_code.strip()
        
        # Encode the mermaid code for URL
        encoded_code = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
        
        # Use mermaid.ink API
        url = f"https://mermaid.ink/img/{encoded_code}"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.content
        else:
            print(f"Mermaid API error: {response.status_code}")
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
        
        # Use PlantUML's specific encoding method
        def plantuml_encode(text):
            """Encode text for PlantUML server using their specific method"""
            import zlib
            
            # Compress using zlib
            compressed = zlib.compress(text.encode('utf-8'), 9)[2:-4]  # Remove zlib headers
            
            # Convert to PlantUML's base64-like encoding
            def encode64(data):
                """PlantUML's custom base64 encoding"""
                chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
                result = ""
                i = 0
                while i < len(data):
                    if i + 2 < len(data):
                        c1, c2, c3 = data[i], data[i+1], data[i+2]
                        result += chars[c1 >> 2]
                        result += chars[((c1 & 0x3) << 4) | (c2 >> 4)]
                        result += chars[((c2 & 0xF) << 2) | (c3 >> 6)]
                        result += chars[c3 & 0x3F]
                        i += 3
                    elif i + 1 < len(data):
                        c1, c2 = data[i], data[i+1]
                        result += chars[c1 >> 2]
                        result += chars[((c1 & 0x3) << 4) | (c2 >> 4)]
                        result += chars[(c2 & 0xF) << 2]
                        i += 2
                    else:
                        c1 = data[i]
                        result += chars[c1 >> 2]
                        result += chars[(c1 & 0x3) << 4]
                        i += 1
                return result
            
            return encode64(compressed)
        
        encoded = plantuml_encode(plantuml_code)
        
        # Use PlantUML server API with correct encoding
        url = f"http://www.plantuml.com/plantuml/png/~1{encoded}"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.content
        else:
            print(f"PlantUML API error: {response.status_code}")
            # Fallback: try a simple approach
            simple_encoded = urllib.parse.quote(plantuml_code)
            fallback_url = f"http://www.plantuml.com/plantuml/png/{simple_encoded}"
            fallback_response = requests.get(fallback_url, timeout=10)
            if fallback_response.status_code == 200:
                return fallback_response.content
            return None
    except Exception as e:
        print(f"Error generating PlantUML diagram: {e}")
        return None

def extract_diagram_syntax(analysis_text):
    """Extract Mermaid or PlantUML syntax from Gemini analysis"""
    mermaid_patterns = [
        r'```mermaid\n(.*?)\n```',
        r'```\n(graph.*?)\n```',
        r'```\n(flowchart.*?)\n```',
        r'(graph\s+\w+.*?)(?=\n\n|\n[A-Z]|\Z)',
        r'(flowchart\s+\w+.*?)(?=\n\n|\n[A-Z]|\Z)'
    ]
    
    plantuml_patterns = [
        r'```plantuml\n(.*?)\n```',
        r'```\n(@startuml.*?@enduml)\n```',
        r'(@startuml.*?@enduml)',
        r'```uml\n(.*?)\n```'
    ]
    
    # Try to find Mermaid syntax
    for pattern in mermaid_patterns:
        match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
        if match:
            return 'mermaid', match.group(1).strip()
    
    # Try to find PlantUML syntax
    for pattern in plantuml_patterns:
        match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
        if match:
            return 'plantuml', match.group(1).strip()
    
    return None, None

def display_generated_diagram(image_data, diagram_type):
    """Display the generated diagram in a new window"""
    try:
        # Convert image data to numpy array
        import numpy as np
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

def process_with_gemini(image_path):
    """Process the saved image with Gemini API to convert sketch to diagram"""
    print("Processing image with Gemini API...")
    
    # Load and prepare the image
    img = PIL.Image.open(image_path)
    
    # Create a prompt for diagram conversion
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
        
        # Save the analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_filename = f"saved_drawings/analysis_{timestamp}.txt"
        
        with open(analysis_filename, 'w') as f:
            f.write("=== GEMINI DIAGRAM ANALYSIS ===\n\n")
            f.write(response.text)
        
        print(f"Analysis saved as: {analysis_filename}")
        print("\n=== GEMINI ANALYSIS ===")
        print(response.text)
        print("=" * 50)
        
        # Extract and generate diagram automatically
        diagram_type, diagram_syntax = extract_diagram_syntax(response.text)
        
        if diagram_type and diagram_syntax:
            print(f"\n🎯 Found {diagram_type} syntax! Generating visual diagram...")
            
            if diagram_type == 'mermaid':
                diagram_image = generate_mermaid_diagram(diagram_syntax)
            elif diagram_type == 'plantuml':
                diagram_image = generate_plantuml_diagram(diagram_syntax)
            
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
            print("ℹ️  No diagram syntax found in analysis")
        
    except Exception as e:
        print(f"Error generating content with Gemini: {e}")

def detect_toolbar_selection(x, y):
    """Detect if user is selecting an item in the professional toolbar"""
    global current_color_index, current_tool_index, canvas, save_button_clicked
    
    # Only detect selections if in toolbar area
    if y >= toolbar_height:
        return False
    
    # Match the professional toolbar layout coordinates
    section_margin = 25
    button_height = 50
    button_spacing = 12
    start_y = (toolbar_height - button_height) // 2
    current_x = section_margin
    
    # === COLOR SELECTION ===
    color_size = 45  # Same as in create_toolbar
    color_spacing = 10  # Same as in create_toolbar
    color_start_y = start_y + 2  # Same as in create_toolbar
    
    for i in range(len(colors)):
        color_x = current_x + i * (color_size + color_spacing)
        color_end_x = color_x + color_size
        color_end_y = color_start_y + color_size
        
        if color_x <= x <= color_end_x and color_start_y <= y <= color_end_y:
            if current_tool_index == tools.index("Eraser"):  # Switch back to previous tool if eraser was selected
                current_tool_index = 0
            current_color_index = i
            return True
    
    current_x += len(colors) * (color_size + color_spacing) + section_margin * 2
    
    # === TOOL SELECTION ===
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
    
    # === ACTION BUTTONS ===
    action_button_width = 80
    
    # Clear All button (first)
    clear_x = current_x
    clear_end_x = clear_x + action_button_width
    clear_end_y = start_y + button_height
    
    if clear_x <= x <= clear_end_x and start_y <= y <= clear_end_y:
        # Clear the canvas (back to white)
        canvas = np.full((WHITEBOARD_HEIGHT, WHITEBOARD_WIDTH, 3), 255, dtype=np.uint8)
        print("Canvas cleared!")
        return True
    
    # Save button (second)
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
    
    # Get relevant finger landmarks
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]  # First joint of index finger
    middle_tip = hand_landmarks.landmark[12]
    
    # Convert to screen coordinates
    index_tip_y = int(index_tip.y * WHITEBOARD_HEIGHT)
    index_pip_y = int(index_pip.y * WHITEBOARD_HEIGHT)
    middle_tip_y = int(middle_tip.y * WHITEBOARD_HEIGHT)
    
    # Detect drawing gesture: index finger up, middle finger down
    drawing_gesture = (index_tip_y < index_pip_y) and (middle_tip_y > index_pip_y)
    
    return drawing_gesture, (int(index_tip.x * WHITEBOARD_WIDTH), int(index_tip.y * WHITEBOARD_HEIGHT))

def main():
    global drawing_mode, prev_points, canvas, start_x, start_y, preview_shape, gesture_start_time, shape_started, save_button_clicked
    
    print("Virtual Painter Enhanced - Controls:")
    print("- Point with index finger to draw")
    print("- Keep middle finger down while drawing")
    print("- Select colors and tools from toolbar")
    print("- Tools: Freehand, Line, Rectangle, Circle, Eraser")
    print("- Use Save button to save and analyze with Gemini")
    print("- Use Clear All button to clear canvas")
    print("- Press 'q' to quit")
    
    if not GEMINI_AVAILABLE:
        print("\nNote: Gemini API not configured. Set GOOGLE_API_KEY environment variable for AI analysis.")
    
    while True:
        # Read frame from webcam
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        # Flip image horizontally for selfie-view
        img = cv2.flip(img, 1)
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process hand landmarks
        results = hands.process(img_rgb)
        
        # Create the main display combining whiteboard and camera
        # Start with whiteboard
        whiteboard_display = canvas.copy()
        
        # Create and overlay toolbar on whiteboard
        toolbar = create_toolbar()
        whiteboard_display[0:toolbar_height, 0:WHITEBOARD_WIDTH] = toolbar
        
        # Resize camera feed for preview
        camera_preview = cv2.resize(img, (CAMERA_WIDTH, CAMERA_HEIGHT))
        
        # Create combined display
        combined_display = np.full((WHITEBOARD_HEIGHT, TOTAL_WIDTH, 3), 240, dtype=np.uint8)  # Light gray background
        combined_display[0:WHITEBOARD_HEIGHT, 0:WHITEBOARD_WIDTH] = whiteboard_display
        # Position camera lower on the right side
        combined_display[CAMERA_Y_OFFSET:CAMERA_Y_OFFSET+CAMERA_HEIGHT, WHITEBOARD_WIDTH:TOTAL_WIDTH] = camera_preview
        
        # Handle save button click
        if save_button_clicked:
            save_drawing()
        
        # Initialize gesture state
        gesture_detected = False
        current_point = None
        
        # Process hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on camera preview (scale coordinates for smaller preview)
                scaled_landmarks = []
                for lm in hand_landmarks.landmark:
                    scaled_x = int(lm.x * CAMERA_WIDTH)
                    scaled_y = int(lm.y * CAMERA_HEIGHT)
                    scaled_landmarks.append([scaled_x, scaled_y])
                
                # Draw landmarks on camera preview manually
                for i, (x, y) in enumerate(scaled_landmarks):
                    cv2.circle(camera_preview, (x, y), 3, (0, 255, 0), -1)
                
                # Get gesture state and finger position (for whiteboard)
                gesture_detected, current_point = get_gesture_state(hand_landmarks)
                
                # ALWAYS show pointer based on index finger position
                index_tip = hand_landmarks.landmark[8]
                finger_point = (int(index_tip.x * WHITEBOARD_WIDTH), int(index_tip.y * WHITEBOARD_HEIGHT))
                
                # Show pointer with color based on state
                if drawing_mode:
                    # Green pointer when drawing
                    cv2.circle(whiteboard_display, finger_point, 12, (0, 255, 0), -1)
                    cv2.circle(whiteboard_display, finger_point, 15, (0, 255, 0), 2)  # Outer ring
                elif gesture_detected:
                    # Yellow pointer when gesture detected but not drawing yet
                    cv2.circle(whiteboard_display, finger_point, 12, (0, 255, 255), -1)
                    cv2.circle(whiteboard_display, finger_point, 15, (0, 255, 255), 2)
                else:
                    # Red pointer when not drawing
                    cv2.circle(whiteboard_display, finger_point, 10, (0, 0, 255), -1)
                    cv2.circle(whiteboard_display, finger_point, 13, (0, 0, 255), 2)
                
                # Handle gesture-based interactions
                if gesture_detected:
                    # Handle toolbar selection
                    if detect_toolbar_selection(current_point[0], current_point[1]):
                        gesture_start_time = 0  # Reset gesture timer when selecting from toolbar
                        drawing_mode = False
                        shape_started = False
                        prev_points = []
                        continue
                    
                    # Gesture timing logic
                    if not drawing_mode:
                        if gesture_start_time == 0:
                            gesture_start_time = time.time()
                        elif time.time() - gesture_start_time >= GESTURE_DELAY:
                            drawing_mode = True
                            if current_tool_index in [1, 2, 3] and not shape_started:  # Line, Rectangle or Circle
                                # Clamp start coordinates to whiteboard boundaries
                                start_x, start_y = clamp_to_whiteboard(current_point[0], current_point[1])
                                shape_started = True
                            prev_points = [current_point]
                    
                    # Drawing logic
                    if drawing_mode and current_point[1] > toolbar_height:
                        # Clamp current point to whiteboard boundaries
                        clamped_point = clamp_to_whiteboard(current_point[0], current_point[1])
                        
                        if current_tool_index == tools.index("Eraser"):
                            # Draw white circle for eraser (erase to white background)
                            cv2.circle(canvas, clamped_point, eraser_thickness, (255, 255, 255), -1)
                        elif current_tool_index == 0:  # Freehand
                            prev_points.append(clamped_point)
                            if len(prev_points) > MAX_POINTS:
                                prev_points.pop(0)
                            
                            if len(prev_points) >= 2:
                                # Draw multiple connecting lines for smoother appearance
                                for i in range(1, len(prev_points)):
                                    cv2.line(canvas, prev_points[i-1], prev_points[i], colors[current_color_index], brush_thickness)
                                
                                # Also draw circles at each point to fill gaps
                                cv2.circle(canvas, clamped_point, brush_thickness//2, colors[current_color_index], -1)
                        
                        elif current_tool_index in [1, 2, 3] and shape_started:  # Line, Rectangle or Circle
                            # Create preview shape
                            preview_shape = canvas.copy()
                            
                            # Clamp start point as well
                            clamped_start = clamp_to_whiteboard(start_x, start_y)
                            
                            if current_tool_index == 1:  # Line
                                cv2.line(preview_shape, clamped_start, clamped_point, colors[current_color_index], brush_thickness)
                            elif current_tool_index == 2:  # Rectangle
                                # Make sure rectangle coordinates are properly ordered
                                top_left = (min(clamped_start[0], clamped_point[0]), min(clamped_start[1], clamped_point[1]))
                                bottom_right = (max(clamped_start[0], clamped_point[0]), max(clamped_start[1], clamped_point[1]))
                                cv2.rectangle(preview_shape, top_left, bottom_right, colors[current_color_index], brush_thickness)
                            else:  # Circle
                                radius = int(calculate_distance(clamped_start, clamped_point))
                                # Clamp radius to stay within boundaries
                                clamped_radius = clamp_circle_radius(clamped_start[0], clamped_start[1], radius)
                                cv2.circle(preview_shape, clamped_start, clamped_radius, colors[current_color_index], brush_thickness)
                else:
                    # Handle gesture release
                    if drawing_mode:
                        if current_tool_index in [1, 2, 3] and preview_shape is not None and shape_started:
                            canvas = preview_shape.copy()
                            preview_shape = None
                            shape_started = False
                        drawing_mode = False
                        gesture_start_time = 0
                        prev_points = []
        
        # Update combined display with latest whiteboard and camera
        display_canvas = preview_shape if preview_shape is not None else canvas
        
        # Update the whiteboard area in combined display
        combined_display[0:WHITEBOARD_HEIGHT, 0:WHITEBOARD_WIDTH] = whiteboard_display
        
        # If there's a preview shape, overlay it on the whiteboard area
        if preview_shape is not None:
            combined_display[toolbar_height:WHITEBOARD_HEIGHT, 0:WHITEBOARD_WIDTH] = preview_shape[toolbar_height:WHITEBOARD_HEIGHT, 0:WHITEBOARD_WIDTH]
        
        # Update camera preview area (positioned lower)
        combined_display[CAMERA_Y_OFFSET:CAMERA_Y_OFFSET+CAMERA_HEIGHT, WHITEBOARD_WIDTH:TOTAL_WIDTH] = camera_preview
        
        # Show the result
        cv2.imshow("Virtual Whiteboard Painter", combined_display)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 