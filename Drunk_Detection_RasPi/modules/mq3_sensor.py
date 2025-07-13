# modules/mq3_sensor.py

import serial
import time

def initialize_serial(port, baudrate):
    try:
        ser = serial.Serial(port, baudrate, timeout=2)
        time.sleep(2)
        print("Serial connection established successfully!")
        return ser
    except Exception as e:
        print(f"Error initializing serial: {e}")
        return None

def read_mq3(ser):
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        data = ser.readline().decode().strip()
        return int(data) if data.isdigit() else None
    except Exception as e:
        print(f"Error reading MQ3 data: {e}")
        return None



def send_command(ser, command):
    try:
        ser.write(command.encode())
    except Exception as e:
        print(f"Error sending command: {e}")
