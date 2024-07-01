import os
import logging
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from dotenv import load_dotenv
import numpy as np
import shutil
from ultralytics import YOLO
from handlers import (
    start,
    button,
    detection,
    handle_conf_selection,
    set_conf,
    handle_iou_selection,
    set_iou,
    handle_class_selection,
    set_class
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file (for local development)
load_dotenv()
TOKEN = os.environ.get('TOKEN')

if TOKEN is None:
    logger.error("No TOKEN found in environment variables")
    raise ValueError("No TOKEN found in environment variables")

model = YOLO("yolov8n.pt")
CONF = 0.25
IOU = 0.7
Y_CLASS = np.arange(0, 80)

def main():
    logger.info("Starting bot...")
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.PHOTO, detection, block=False))
    application.run_polling()
    logger.info("Bot started successfully")

if __name__ == "__main__":
    main()
