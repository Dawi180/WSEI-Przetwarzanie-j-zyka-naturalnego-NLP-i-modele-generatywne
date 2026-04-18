import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from config import TELEGRAM_TOKEN
import bot_commands

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def podsluch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Przechwytuje absolutnie każdą wiadomość, zanim bot cokolwiek z nią zrobi."""
    if update.message and update.message.text:
        print(f"\n[👀 PODSŁUCH] Złapałem wiadomość: '{update.message.text}'")

def main():
    print("Uruchamianie bota z radarem diagnostycznym...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # RADAR (group=-1 sprawia, że uruchomi się pierwszy, niezależnie od komend)
    app.add_handler(MessageHandler(filters.TEXT, podsluch), group=-1)

    # Rejestracja komend Lab 3
    app.add_handler(CommandHandler("help", bot_commands.help_command))
    app.add_handler(CommandHandler("add_sentiment", bot_commands.add_sentiment_command))
    app.add_handler(CommandHandler("sentiment", bot_commands.sentiment_command))
    app.add_handler(CommandHandler("train", bot_commands.train_command))
    app.add_handler(CommandHandler("models", bot_commands.models_command))
    app.add_handler(CommandHandler("compare", bot_commands.compare_command))

    # Rejestracja komend Lab 1 i 2
    app.add_handler(CommandHandler("task", bot_commands.task_command))
    app.add_handler(CommandHandler("full_pipeline", bot_commands.full_pipeline_command))
    app.add_handler(CommandHandler("classifier", bot_commands.classifier_command))
    app.add_handler(CommandHandler("stats", bot_commands.stats_command))
    app.add_handler(CommandHandler("classify", bot_commands.classify_lab2_command))

    print("Bot jest gotowy! Nasłuchuję...")
    app.run_polling()

if __name__ == '__main__':
    main()