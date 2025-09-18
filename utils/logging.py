import os
from datetime import datetime
import threading
import json

file_lock = threading.Lock()

def log_to_file(log_dir, message, show_time = False):
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with file_lock:
        with open(log_dir, 'a', encoding='utf-8') as f:
            if show_time:
                f.write(f"{timestamp} - {message}\n")
            else:
                f.write(f"{message}\n")

def reset_file(log_dir):
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    with open(log_dir, 'w', encoding='utf-8') as f:
        f.write("")