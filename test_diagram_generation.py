#!/usr/bin/env python3
"""Test script to demonstrate diagram generation functionality"""

import cv2
import numpy as np
import requests
import base64
import re
from datetime import datetime

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

def display_generated_diagram(image_data, diagram_type):
    """Display the generated diagram in a new window"""
    try:
        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        diagram_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if diagram_img is not None:
            # Display the diagram
            cv2.imshow(f"Test {diagram_type.title()} Diagram", diagram_img)
            print(f"✨ {diagram_type.title()} diagram displayed! Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return True
        else:
            print("Failed to decode generated diagram")
            return False
    except Exception as e:
        print(f"Error displaying diagram: {e}")
        return False

def test_diagram_generation():
    """Test the diagram generation with sample Mermaid code"""
    print("Testing diagram generation...")
    
    # Sample Mermaid flowchart
    sample_mermaid = """
    graph TD
        A[Start] --> B{Is it working?}
        B -->|Yes| C[Great!]
        B -->|No| D[Fix it]
        D --> B
        C --> E[End]
    """
    
    print("Generating diagram from Mermaid syntax...")
    diagram_image = generate_mermaid_diagram(sample_mermaid)
    
    if diagram_image:
        # Save the generated diagram
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_diagram_{timestamp}.png"
        
        with open(filename, 'wb') as f:
            f.write(diagram_image)
        
        print(f"📊 Test diagram saved as: {filename}")
        
        # Display the diagram
        success = display_generated_diagram(diagram_image, 'mermaid')
        
        if success:
            print("✅ Diagram generation test successful!")
        else:
            print("❌ Failed to display diagram")
    else:
        print("❌ Failed to generate diagram")

if __name__ == "__main__":
    test_diagram_generation() 