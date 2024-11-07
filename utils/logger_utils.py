import logging
import os
import datetime

def setup_logger(logs_dir):
    # logs_dir = '/home/val/Documents/SSL/logs'
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger('RunLogger')
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        timestamp = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
        file_handler = logging.FileHandler(os.path.join(logs_dir, f'run_{timestamp}.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
