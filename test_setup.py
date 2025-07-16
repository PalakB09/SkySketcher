#!/usr/bin/env python3
"""Test script to verify all dependencies are working"""

import sys

def test_imports():
    """Test all required imports"""
    try:
        print("Testing imports...")
        
        import cv2
        print("✅ OpenCV imported successfully")
        
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import websockets
        print("✅ WebSockets imported successfully")
        
        import google.generativeai as genai
        print("✅ Google Generative AI imported successfully")
        
        import PIL.Image
        print("✅ Pillow (PIL) imported successfully")
        
        import requests
        print("✅ Requests imported successfully")
        
        print("\n🎉 All dependencies imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_camera():
    """Test camera access"""
    try:
        print("\nTesting camera access...")
        import cv2
        
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera access successful")
            cap.release()
            return True
        else:
            print("❌ Camera not accessible")
            return False
            
    except Exception as e:
        print(f"❌ Camera test error: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe hands model"""
    try:
        print("\nTesting MediaPipe hands...")
        import mediapipe as mp
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False,
                              max_num_hands=1,
                              min_detection_confidence=0.7,
                              min_tracking_confidence=0.7)
        print("✅ MediaPipe hands model initialized")
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe test error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Collaborative Air Canvas Setup Test ===\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Some imports failed. Please run: pip3 install -r requirements.txt")
        sys.exit(1)
    
    # Test camera
    camera_ok = test_camera()
    
    # Test MediaPipe
    mediapipe_ok = test_mediapipe()
    
    print("\n=== Test Summary ===")
    print("✅ Dependencies: All imported successfully")
    print(f"{'✅' if camera_ok else '❌'} Camera: {'Working' if camera_ok else 'Not accessible'}")
    print(f"{'✅' if mediapipe_ok else '❌'} MediaPipe: {'Working' if mediapipe_ok else 'Error'}")
    
    if camera_ok and mediapipe_ok:
        print("\n🎉 Setup complete! You can now run:")
        print("1. python3 websocket_server.py (in one terminal)")
        print("2. python3 collaborative_airCanvas.py (in another terminal)")
        print("\nFor multiple users, run step 2 in multiple terminals!")
    else:
        print("\n⚠️  Some components may not work properly. Check the errors above.")

if __name__ == "__main__":
    main() 