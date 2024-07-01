from telegram.ext import CallbackQueryHandler, CommandHandler, MessageHandler, filters
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import numpy as np
import os
import shutil
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
CONF = 0.25
IOU = 0.7
Y_CLASS = np.arange(0, 80)

def create_keyboard(buttons):
    return [[InlineKeyboardButton(text, callback_data=data) for text, data in buttons]]

async def show_start_keyboard(message):
    keyboard = create_keyboard([("Set Conf", "1"), ("Set IOU", "2"), ("Set Class", "3")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await message.reply_text('Set settings or Send image.', reply_markup=reply_markup)

async def start(update, context):
    keyboard = create_keyboard([("Set Conf", "1"), ("Set IOU", "2"), ("Set Class", "3")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('This bot can detect objects on image. Set settings or Send image.', reply_markup=reply_markup)

async def button(update, context):
    query = update.callback_query
    await query.answer()
    if query.data == '1':
        await handle_conf_selection(query)
    elif query.data == '2':
        await handle_iou_selection(query)
    elif query.data == '3':
        await handle_class_selection(query)
    elif query.data.startswith('conf'):
        await set_conf(query)
    elif query.data.startswith('iou'):
        await set_iou(query)
    elif query.data.startswith('class'):
        await set_class(query)

async def handle_conf_selection(query):
    keyboard = create_keyboard([("0.1", "conf0.1"), ("0.5", "conf0.5"), ("0.9", "conf0.9")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.message.edit_text('Set desired lvl of CONF', reply_markup=reply_markup)

async def set_conf(query):
    global CONF
    CONF = float(query.data.replace('conf', ''))
    await query.message.edit_text(f'CONF = {CONF}')
    await show_start_keyboard(query.message)

async def handle_iou_selection(query):
    keyboard = create_keyboard([("0.01", "iou0.01"), ("0.5", "iou0.5"), ("0.99", "iou0.99")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.message.edit_text('Set desired lvl of IOU', reply_markup=reply_markup)

async def set_iou(query):
    global IOU
    IOU = float(query.data.replace('iou', ''))
    await query.message.edit_text(f'IOU = {IOU}')
    await show_start_keyboard(query.message)

async def handle_class_selection(query):
    keyboard = create_keyboard([("Person", "class0"), ("Car", "class2"), ("Cat", "class15"), ("Dog", "class16"), ("All", "classALL")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.message.edit_text('Set desired CLASS', reply_markup=reply_markup)

async def set_class(query):
    global Y_CLASS
    if query.data != 'classALL':
        Y_CLASS = int(query.data.replace('class', ''))
    else:
        Y_CLASS = np.arange(0, 80)
    await query.message.edit_text(f'CLASS = {Y_CLASS}')

async def detection(update, context):
    try:
        shutil.rmtree('images')
        shutil.rmtree('runs')
    except:
        pass
    my_message = await update.message.reply_text('Image received, processing...')
    new_file = await update.message.photo[-1].get_file()
    os.makedirs('images', exist_ok=True)
    image_name = str(new_file['file_path']).split('/')[-1]
    image_path = os.path.join('images', image_name)
    await new_file.download_to_drive(image_path)
    model.predict(image_path, save=True, conf=CONF, iou=IOU, classes=Y_CLASS)
    await context.bot.deleteMessage(message_id=my_message.message_id, chat_id=update.message.chat_id)
    await update.message.reply_text('Detection completed')
    await update.message.reply_photo(f'runs/detect/predict/{image_name}')
    await show_start_keyboard(update.message)
