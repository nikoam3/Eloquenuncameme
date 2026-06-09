import telebot
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

telegram = telebot.TeleBot(TELEGRAM_TOKEN)

def enviar(mensaje: str, modo: str = 'Markdown'):
    try:
        telegram.send_message(
            TELEGRAM_CHAT_ID, 
            mensaje,
            parse_mode=modo
        )
    except Exception as e:
        print(f"Error Telegram: {e}")