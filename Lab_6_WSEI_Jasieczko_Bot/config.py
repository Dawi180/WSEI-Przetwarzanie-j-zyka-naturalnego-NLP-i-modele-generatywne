import os
from dotenv import load_dotenv

# Wczytanie zmiennych środowiskowych z pliku .env
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("Brak tokena! Upewnij się, że masz plik .env z TELEGRAM_TOKEN.")

# Ścieżki do folderów (wymagane w Lab 3)
MODELS_DIR = "models"
PLOTS_DIR_LAB3 = "lab3plots"
DATASET_FILE = "sentiment_dataset.csv"

# Automatyczne tworzenie folderów dla modeli i wykresów z Lab 3
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR_LAB3, exist_ok=True)