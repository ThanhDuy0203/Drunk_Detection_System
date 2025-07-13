# config.py
import os
import json

MODEL_PATH = "/home/pluto/Documents/Drunk_Detection/models/tflife_best_model.tflife"
TELEGRAM_TOKEN = '8103302211:AAG-SWWfbp5u7PJFNUl8A8a8nsjyG4LHEEw'
TELEGRAM_CHAT_ID = '-4752733894'
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUDRATE = 9600
MQ3_THRESHOLD = 400
DRUNK_DETECTION_SECONDS = 40
FRAME_INTERVAL = 5


DRIVERS_JSON_PATH = "drivers.json"
DRIVERS = []

if os.path.exists(DRIVERS_JSON_PATH):
    try:
        with open(DRIVERS_JSON_PATH, 'r', encoding='utf-8') as f:
            DRIVERS = json.load(f)
    except Exception as e:
        print(f"Error loading drivers.json: {e}")

if DRIVERS:
    CURRENT_DRIVER = DRIVERS[0]  
else:
    print("Warning: No drivers found, using default driver")
    CURRENT_DRIVER = {"id": "UNKNOWN", "name": "Unknown Driver", "plate": "UNKNOWN"}
