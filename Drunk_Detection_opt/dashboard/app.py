from flask import Flask, render_template, request
import pandas as pd
import os
from dataclasses import dataclass

app = Flask(__name__)

# Dataclass mapping dòng CSV
@dataclass
class WarningLog:
    time: str
    driver_id: str
    driver_name: str
    vehicle_plate: str
    mq3_value: int
    photo_path: str = ''
    photo_url: str = None

    def attach_photo_url(self):
        if self.photo_path and os.path.exists(f'static/images/{self.photo_path}'):
            self.photo_url = f'/static/images/{self.photo_path}'
        else:
            self.photo_url = None

# Hàm đọc dữ liệu từ CSV và convert sang object
def load_logs_from_csv(file_path=r'D:\FPTUniversity\Capstone_Project\Drunk_Detection\dashboard\warning.csv'):
    if not os.path.exists(file_path):
        return []

    df = pd.read_csv(file_path)
    logs = []

    for _, row in df.iterrows():
        log = WarningLog(
            time=row.get('time', ''),
            driver_id=str(row.get('driver_id', '')),
            driver_name=row.get('driver_name', ''),
            vehicle_plate=row.get('vehicle_plate', ''),
            mq3_value=int(row.get('mq3_value', 0)),
            photo_path=row.get('photo_path', '')
        )
        log.attach_photo_url()
        logs.append(log)

    return logs

# Route chính
@app.route('/', methods=['GET'])
def index():
    logs = load_logs_from_csv()


    driver_id = request.args.get('driver_id', '')
    if driver_id:
        logs = [log for log in logs if log.driver_id == driver_id]
    return render_template('index.html', logs=logs)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
