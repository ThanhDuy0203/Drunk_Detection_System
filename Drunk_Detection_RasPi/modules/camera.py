# modules/camera.py

from picamera2 import Picamera2
import cv2

def initialize_camera():
    try:
        picam = Picamera2()
        config = picam.create_preview_configuration(main={"size": (640, 480)})
        picam.configure(config)
        picam.start()
        return picam
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return None

def capture_frame(picam):
    try:
        image = picam.capture_array()
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error capturing frame: {e}")
        return None
