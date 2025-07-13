# main.py
import time
import sys
import cv2
from modules.camera import initialize_camera, capture_frame
from modules.mq3_sensor import initialize_serial, read_mq3, send_command
from modules.image_processing import load_tflite_model, predict_image, preprocess_image
from modules.telegram_bot import send_telegram_message, format_warning_message
from modules.logger import log_warning
from config import (
    MODEL_PATH, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SERIAL_PORT, SERIAL_BAUDRATE,
    MQ3_THRESHOLD, DRUNK_DETECTION_SECONDS, FRAME_INTERVAL, CURRENT_DRIVER, DRIVERS
)

def process_frame(frame, model, drunk_frame, mq3_value, ser, driver):
    frame = frame[::-1].copy()
    processed_frame = preprocess_image(frame)

    alcohol_status = "Alcohol Detected" if mq3_value and mq3_value > MQ3_THRESHOLD else "No Alcohol"
    image_status = "Not Drunk"
    if processed_frame is not None:
        image_status = predict_image(model, processed_frame)

    final_status = "Drunk" if (image_status == "Drunk" or alcohol_status == "Alcohol Detected") else "Not Drunk"

    if final_status == "Drunk":
        drunk_frame[0] += 1
        max_frames = int(DRUNK_DETECTION_SECONDS / FRAME_INTERVAL)
        if drunk_frame[0] >= max_frames:
            frame_to_save = cv2.flip(frame, 0)
            frame_to_save = cv2.resize(frame_to_save, (320, 240), interpolation=cv2.INTER_AREA)
            photo_path = f"drunk_face_{int(time.time())}.jpg"
            cv2.imwrite(photo_path, frame_to_save)

            message = format_warning_message(mq3_value, driver['id'], driver['name'], driver['plate'])
            send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message, photo_path)
            log_warning(driver['id'], driver['name'], driver['plate'], mq3_value, photo_path)
            drunk_frame[0] = 0
        send_command(ser, '1')
    else:
        drunk_frame[0] = 0
        send_command(ser, '0')

    print(f"Status: {final_status}, MQ3 Value: {mq3_value if mq3_value else 'N/A'}, Drunk frames: {drunk_frame[0]}")

def cleanup_resources(picam, ser):
    try:
        if picam is not None:
            picam.stop()
        if ser is not None:
            ser.flush()
            ser.close()
        print("Resources cleaned up.")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def main():
    model = load_tflite_model(MODEL_PATH)
    if model is None:
        print("Failed to load model. Exiting.")
        return

    picam = initialize_camera()
    if picam is None:
        print("Failed to initialize camera. Exiting.")
        return

    ser = initialize_serial(SERIAL_PORT, SERIAL_BAUDRATE)
    if ser is None:
        print("Failed to initialize serial communication with Arduino. Exiting.")
        cleanup_resources(picam, None)
        return

    time.sleep(1.0)
    drunk_frame = [0]
    last_time = time.time()

   
    print(f"Loaded {len(DRIVERS)} drivers")
    print(f"Current driver: {CURRENT_DRIVER['id']} - {CURRENT_DRIVER['name']}")

    send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, "**He thong khoi dong thanh cong!**")
    print("System started.")

    try:
        while True:
            frame = capture_frame(picam)
            if frame is None:
                continue

            mq3_value = read_mq3(ser)

            current_time = time.time()
            if current_time - last_time >= FRAME_INTERVAL:
                process_frame(frame, model, drunk_frame, mq3_value, ser, CURRENT_DRIVER)
                last_time = current_time

    except Exception as e:
        print(f"Unexpected error: {e}")
        send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, f"**Loi he thong*: {e}")
    finally:
        cleanup_resources(picam, ser)
        sys.exit(0)

if __name__ == "__main__":
    main()
