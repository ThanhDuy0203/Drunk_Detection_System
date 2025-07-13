# modules/telegram_bot.py

import telegram
from datetime import datetime
from config import DRUNK_DETECTION_SECONDS

def send_telegram_message(token, chat_id, message, photo_path=None):
    try:
        bot = telegram.Bot(token)
        if photo_path:
            
            with open(photo_path, 'rb') as photo:
                bot.send_photo(chat_id=chat_id, photo=photo, caption=message, parse_mode='Markdown')
        else:
            
            bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def format_warning_message(mq3_value, driver_id, driver_name, vehicle_plate):
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = (
        " *DRUNK DRIVER ALERT!!*\n"
        f"- **Thoi gian**: {current_time}\n"
        f"- **ID**: {driver_id}\n"
        f"- **Ho va ten**: {driver_name}\n"
        f"- **Bien so xe**: {vehicle_plate}\n"
        f"- **Chi tiet**: Phat hien say xin lien tuc trong {DRUNK_DETECTION_SECONDS} giay. Gia tri MQ3: {mq3_value if mq3_value else 'N/A'}"
    )
    return message
