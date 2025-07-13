# modules/logger.py
import csv
from datetime import datetime

def log_warning(driver_id, driver_name, vehicle_plate, mq3_value, photo_path):
    with open('warnings.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['time', 'driver_id', 'driver_name', 'vehicle_plate', 'mq3_value', 'photo_path'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), driver_id, driver_name, vehicle_plate, mq3_value, photo_path])
