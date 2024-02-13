import logging
import os
from datetime import datetime

# Correcting the log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "logs")  # Correcting the usage of os.path.join
os.makedirs(log_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
)

if __name__ == "__main__":
    print ('abcd')
    logging.info("logging has started")
