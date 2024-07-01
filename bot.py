import os
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from dotenv import load_dotenv
import numpy as np
import shutil
from ultralytics import YOLO
from bot.handlers import (
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

load_dotenv()
TOKEN = os.environ.get('TOKEN')

def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.PHOTO, detection, block=False))
    application.run_polling()

if __name__ == "__main__":
    main()
