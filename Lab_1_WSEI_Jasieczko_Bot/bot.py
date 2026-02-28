import os
import shlex
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv

# Importujemy nasze własne moduły
import data_handler
import nlp_tools
import ml_model

# --- FUNKCJE POMOCNICZE ---

def parse_args(text):
    """Pomocnicza funkcja do bezpiecznego dzielenia tekstu uwzględniająca cudzysłowy."""
    try:
        # shlex.split świetnie radzi sobie z tekstami typu: /komenda "tekst ze spacjami" "klasa"
        return shlex.split(text)[1:] 
    except ValueError:
        return []

# --- OBSŁUGA KOMEND ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Powitanie i instrukcja."""
    help_text = (
        "Cześć! Jestem botem NLP.\n"
        "Dostępne komendy:\n"
        "/task <nazwa_zadania> \"tekst\" \"klasa\"\n"
        "/full_pipeline \"tekst\" \"klasa\"\n"
        "/classifier \"tekst\"\n"
        "/stats"
    )
    await update.message.reply_text(help_text)

async def task_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa komendy /task."""
    args = parse_args(update.message.text)
    if len(args) != 3:
        await update.message.reply_text("Błąd składni. Użycie: /task <nazwa_zadania> \"tekst\" \"klasa\"")
        return

    task_name, text, text_class = args
    
    # 1. Zapis do bazy
    data_handler.save_record(text, text_class)
    
    # 2. Wykonanie zadania NLP i odpowiedź
    tokens = nlp_tools.tokenize_text(text)
    response_text = ""
    photo_path = None

    if task_name == "tokenize":
        response_text = f"Tokeny: {tokens}"
    elif task_name == "remove_stopwords":
        res = nlp_tools.remove_stopwords(tokens)
        response_text = f"Bez stop words: {res}"
    elif task_name == "lemmatize":
        res = nlp_tools.lemmatize_tokens(tokens)
        response_text = f"Lematyzacja: {res}"
    elif task_name == "stemming":
        res = nlp_tools.stem_tokens(tokens)
        response_text = f"Stemming: {res}"
    elif task_name == "n-grams":
        res = nlp_tools.generate_ngrams(tokens, 2)
        response_text = f"Bigramy: {res}"
    elif task_name == "plot_histogram":
        photo_path = nlp_tools.plot_histogram(tokens)
        response_text = "Wygenerowano histogram długości tokenów."
    elif task_name == "plot_wordcloud":
        photo_path = nlp_tools.plot_wordcloud(text)
        response_text = "Wygenerowano word cloud."
    elif task_name == "stats": # Tymczasowa statystyka z tokenów
         response_text = f"Liczba znaków: {len(text)}, liczba tokenów: {len(tokens)}"
    else:
        response_text = "Nieznane zadanie. Wybierz np.: tokenize, lemmatize, plot_histogram..."

    await update.message.reply_text(response_text)
    
    if photo_path and os.path.exists(photo_path):
        await update.message.reply_photo(photo=open(photo_path, 'rb'))

async def full_pipeline_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa komendy /full_pipeline."""
    args = parse_args(update.message.text)
    if len(args) != 2:
        await update.message.reply_text("Błąd składni. Użycie: /full_pipeline \"tekst\" \"klasa\"")
        return

    full_text, text_class = args
    sentences = nlp_tools.split_sentences(full_text)
    
    await update.message.reply_text(f"Rozpoczynam pełny pipeline dla {len(sentences)} zdań...")

    all_tokens = []
    
    for idx, sentence in enumerate(sentences):
        # Naiwne przypisanie tej samej klasy każdemu zdaniu
        data_handler.save_record(sentence, text_class)
        
        cleaned = nlp_tools.clean_text(sentence)
        tokens = nlp_tools.tokenize_text(cleaned)
        no_stop = nlp_tools.remove_stopwords(tokens)
        lemmas = nlp_tools.lemmatize_tokens(no_stop)
        stems = nlp_tools.stem_tokens(no_stop)
        
        all_tokens.extend(tokens)
        
        raport = (
            f"--- Zdanie {idx+1} ---\n"
            f"Oryginał: {sentence}\n"
            f"Czyszczenie: {cleaned}\n"
            f"Tokeny: {tokens}\n"
            f"Bez stop words: {no_stop}\n"
            f"Lematy: {lemmas}\n"
            f"Stemy: {stems}\n"
        )
        await update.message.reply_text(raport)

    # Reprezentacje BoW i TF-IDF dla całości wprowadzonych zdań
    if len(sentences) > 0:
        bow_vocab, bow_array = nlp_tools.get_bow(sentences)
        tfidf_vocab, tfidf_array = nlp_tools.get_tfidf(sentences)
        await update.message.reply_text("Policzono reprezentacje Bag of Words oraz TF-IDF dla podanego tekstu.")

    # Generowanie wykresów dla całego tekstu
    await update.message.reply_text("Generuję wykresy...")
    
    plots = [
        nlp_tools.plot_bar_chart(all_tokens),
        nlp_tools.plot_histogram(all_tokens),
        nlp_tools.plot_wordcloud(full_text)
    ]
    
    for plot_path in plots:
        if plot_path and os.path.exists(plot_path):
            await update.message.reply_photo(photo=open(plot_path, 'rb'))

async def classifier_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa komendy /classifier."""
    args = parse_args(update.message.text)
    if len(args) != 1:
        await update.message.reply_text("Błąd składni. Użycie: /classifier \"tekst\"")
        return

    text_to_classify = args[0]
    result = ml_model.train_and_predict(text_to_classify)
    await update.message.reply_text(result)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa komendy /stats."""
    data = data_handler.load_data()
    if not data:
        await update.message.reply_text("Baza danych jest pusta. Użyj najpierw /task lub /full_pipeline.")
        return
        
    await update.message.reply_text("Generuję statystyki całego zbioru danych...")
    
    all_text = " ".join([item.get("text", "") for item in data])
    cleaned_text = nlp_tools.clean_text(all_text)
    tokens = nlp_tools.tokenize_text(cleaned_text)
    
    unique_tokens = set(tokens)
    bigrams = set(nlp_tools.generate_ngrams(tokens, 2))
    trigrams = set(nlp_tools.generate_ngrams(tokens, 3))
    
    class_stats = data_handler.get_classes_stats()
    
    stats_text = (
        f"Liczność klas:\n{class_stats}\n\n"
        f"Unikalne tokeny: {len(unique_tokens)}\n"
        f"Unikalne 2-gramy: {len(bigrams)}\n"
        f"Unikalne 3-gramy: {len(trigrams)}\n"
    )
    await update.message.reply_text(stats_text)
    
    plots = [
        nlp_tools.plot_bar_chart(tokens),
        nlp_tools.plot_histogram(tokens),
        nlp_tools.plot_wordcloud(all_text)
    ]
    
    for plot_path in plots:
        if plot_path and os.path.exists(plot_path):
            await update.message.reply_photo(photo=open(plot_path, 'rb'))

# --- MAIN ---

if __name__ == '__main__':
    # Wczytujemy zmienne z pliku .env
    load_dotenv()
    
    # Pobieramy token z ukrytej zmiennej środowiskowej
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    
    # Zabezpieczenie, gdyby plik .env nie istniał lub był pusty
    if not TOKEN:
        raise ValueError("Brak tokena! Upewnij się, że masz plik .env z TELEGRAM_TOKEN.")
    
    print("Uruchamianie bota...")
    app = ApplicationBuilder().token(TOKEN).build()

    # Rejestracja komend
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("task", task_command))
    app.add_handler(CommandHandler("full_pipeline", full_pipeline_command))
    app.add_handler(CommandHandler("classifier", classifier_command))
    app.add_handler(CommandHandler("stats", stats_command))

    print("Bot jest gotowy! Możesz do niego pisać na Telegramie.")
    app.run_polling()