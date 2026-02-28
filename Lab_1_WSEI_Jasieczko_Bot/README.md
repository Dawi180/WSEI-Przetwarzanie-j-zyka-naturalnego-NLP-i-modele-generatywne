# Telegram NLP Bot - Laboratorium 1

Prosty bot do przetwarzania i klasyfikacji wiadomości tekstowych.

## Wymagania
- Python 3.8+
- Utworzone wirtualne środowisko (venv)
- Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

## Instalacja
1. Aktywuj wirtualne środowisko.
2. Zainstaluj wymagane pakiety:
   `pip install python-telegram-bot nltk scikit-learn matplotlib seaborn wordcloud python-dotenv`
3. Utwórz nowy plik o nazwie .env w głównym folderze projektu. "TELEGRAM_TOKEN=Twój_Token_Od_BotFathera"


## Uruchomienie
Wpisz w terminalu:
`python bot.py`