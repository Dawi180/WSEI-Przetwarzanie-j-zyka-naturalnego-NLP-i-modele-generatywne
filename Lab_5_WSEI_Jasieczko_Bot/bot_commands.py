import os
import shlex
import asyncio
from telegram import Update
from telegram.ext import ContextTypes
import re
import sentiment_methods
import dl_models
import comparison

# Importy z Lab 1 i 2
import data_handler
import nlp_tools
import ml_model
import lab2_experiments

# Importy Lab 4
import nlp_advanced
import translation_tools
import knowledge_graph
import llm_tools
import asyncio

# Importy Lab 5
import llm_agent
import vision_tools
import os

# --- FUNKCJE POMOCNICZE ---
def parse_args(text):
    try:
        return shlex.split(text)[1:] 
    except ValueError:
        return []

# ==========================================
#         KOMENDY Z LAB 5 (NOWE)
# ==========================================

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
"🤖 **Witaj w Super-Bocie NLP (Lab 5)!**\n\n"
        "🌟 **Nowości (Agent & Vision):**\n"
        "🧠 `/agent <zapytanie>` - Inteligentny asystent z dostępem do narzędzi (Pogoda, Internet, Kalkulator).\n"
        "📸 **Wyślij zdjęcie** (z opcjonalnym podpisem), aby model Vision je przeanalizował!\n\n"
        "🔍 `/ner`, `/translate`, `/summarize`, `/analyze_entities` (Z Lab 4)\n"
        "🔍 `/ner method=<spacy|stanza> text=\"tekst\"` - Wykrywa osoby, miejsca, organizacje.\n"
        "🌍 `/translate text=\"tekst\" target_lang=<en|pl|de|fr>` - Darmowe tłumaczenie AI.\n"
        "🧠 `/summarize text=\"tekst\" type=<abstractive|extractive|bullets> length=<short|medium|long>` - Streszczenie używając Ollama.\n"
        "🔗 `/analyze_entities text=\"tekst\" link=<true|false>` - Tworzy Graf Wiedzy (PNG) i szuka w Wikipedii.\n"
        "🗣️ `/language_detect text=\"tekst\"` - Wykrywa język.\n\n"
        "🛠️ **Stare komendy (Lab 3):**\n"
        "📝 `/add_sentiment \"tekst\" \"etykieta\"` - Dodaje tekst do datasetu.\n"
        "🧠 `/sentiment method=<metoda> text=\"tekst\"` - Analiza sentymentu.\n"
        "🚀 `/train model=<model> dataset=<dataset>` - Trenuje sieć neuronową.\n"
        "📊 `/compare dataset=<dataset> methods=<metody>` - Porównuje metody.\n"
        "📂 `/models` - Wyświetla listę wytrenowanych modeli.\n\n"
        "🛠️ **Stare komendy (Lab 1 & 2):**\n"
        "`/task`, `/full_pipeline`, `/classifier`, `/stats`, `/classify`."
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def add_sentiment_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa komendy /add_sentiment "tekst" "etykieta" """
    args = parse_args(update.message.text)
    if len(args) != 2:
        await update.message.reply_text("❌ Błąd składni. Użycie: `/add_sentiment \"tekst\" \"etykieta\"`", parse_mode="Markdown")
        return

    text, label = args
    valid_labels = ["pozytywny", "neutralny", "negatywny"]
    
    if label.lower() not in valid_labels:
        await update.message.reply_text(f"❌ Błąd! Dozwolone etykiety to: {', '.join(valid_labels)}")
        return

    # Zapisujemy do CSV
    data_handler.append_to_csv(text, label.lower())
    await update.message.reply_text(f"✅ Zapisano pomyślnie w `sentiment_dataset.csv`!\n\nTekst: {text}\nEtykieta: {label}", parse_mode="Markdown")

async def sentiment_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa komendy /sentiment method=<metoda> text="tekst" """
    message = update.message.text
    
    # Używamy wyrażeń regularnych, by wyciągnąć metodę i tekst z cudzysłowów
    method_match = re.search(r'method=(\w+)', message)
    text_match = re.search(r'text="([^"]+)"', message)

    if not method_match or not text_match:
        await update.message.reply_text("❌ Błąd składni. Użycie: `/sentiment method=<metoda> text=\"tekst\"`", parse_mode="Markdown")
        return

    method = method_match.group(1).lower()
    text = text_match.group(1)

    # Uruchamiamy odpowiedni model
    if method == "rule":
        result = sentiment_methods.analyze_rule_based(text)
    elif method == "textblob":
        result = sentiment_methods.analyze_textblob(text)
    elif method == "transformer":
        res_label, score = sentiment_methods.analyze_transformer(text)
        result = f"{res_label} (Pewność: {score:.2f})"
    elif method == "stanza":
        result = sentiment_methods.analyze_stanza(text)
    elif method in ["simplernn", "lstm", "gru"]:
        result = dl_models.predict_sentiment(text, method, dataset_name="imdb")
    else:
        await update.message.reply_text(f"🚧 Metoda `{method}` nie istnieje. Dostępne: rule, textblob, transformer, stanza, simplernn, lstm, gru", parse_mode="Markdown")
        return

    # Sformatowana odpowiedź dla użytkownika
    raport = (
        f"🤖 **Wynik analizy sentymentu**\n"
        f"Model: `{method}`\n"
        f"Predykcja: **{result}**"
    )
    await update.message.reply_text(raport, parse_mode="Markdown")

async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa komendy /train z pełnym śledzeniem (debugowaniem)"""
    text = update.message.text
    print(f"\n--- KROK 1: Otrzymano komendę: {text}")
    
    import re
    model_match = re.search(r'model=(\w+)', text)
    dataset_match = re.search(r'dataset=(\w+)', text)

    if not model_match or not dataset_match:
        print("--- KROK 2: Odrzucono - brak parametrów")
        await update.message.reply_text("❌ Użycie: `/train model=<lstm|gru|simplernn> dataset=<imdb|amazon|custom>`", parse_mode="Markdown")
        return

    model_type = model_match.group(1).lower()
    dataset_name = dataset_match.group(1).lower()
    print(f"--- KROK 3: Wyciągnięto zmienne -> model: {model_type}, dataset: {dataset_name}")

    print("--- KROK 4: Wysyłam wiadomość o rozpoczęciu na Telegram...")
    await update.message.reply_text(f"⏳ Rozpoczynam trening sieci **{model_type.upper()}** na zbiorze **{dataset_name}**.\nTo może potrwać kilka minut...", parse_mode="Markdown")
    print("--- KROK 5: Wiadomość wysłana! Przekazuję zadanie do silnika TensorFlow...")

    try:
        model_path, plot_path, final_acc = await asyncio.to_thread(
            dl_models.train_and_save_model, model_type, dataset_name
        )
        print("--- KROK 6: Sukces! Wątek poboczny zwrócił wyniki.")
        
        raport = (
            f"✅ **Trening zakończony sukcesem!**\n\n"
            f"🧠 Model: `{model_type.upper()}`\n"
            f"📂 Zapisano w: `{model_path}`\n"
            f"🎯 Ostateczna dokładność (Val Acc): **{final_acc:.2f}**"
        )
        await update.message.reply_text(raport, parse_mode="Markdown")
        
        if os.path.exists(plot_path):
            await update.message.reply_photo(photo=open(plot_path, 'rb'))
            
    except Exception as e:
        print(f"--- BŁĄD KRYTYCZNY KROK 6: {e}")
        import traceback
        traceback.print_exc()
        await update.message.reply_text(f"❌ Wystąpił błąd podczas treningu: {str(e)}")


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa komendy /models - wyświetla wytrenowane sieci .h5"""
    models = dl_models.get_available_models()
    
    if not models:
        await update.message.reply_text("📂 Katalog modeli jest pusty. Wytrenuj coś komendą `/train`!", parse_mode="Markdown")
        return
        
    text = "📂 **Zapisane modele (.h5):**\n\n"
    for m in models:
        text += f"▪️ `{m}` (Posiada pliki tokenizer/encoder)\n"
        
    await update.message.reply_text(text, parse_mode="Markdown")

async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa komendy /compare dataset=<dataset> methods=<lista_metod>"""
    text = update.message.text
    import re

    # Regex, który łapie WSZYSTKO po 'methods=' aż do końca wiadomości
    dataset_match = re.search(r'dataset=(\w+)', text)
    methods_match = re.search(r'methods=(.+)', text)

    if not dataset_match or not methods_match:
        await update.message.reply_text("❌ Użycie: `/compare dataset=<imdb|amazon|custom> methods=<rule,textblob,lstm...>`", parse_mode="Markdown")
        return

    dataset_name = dataset_match.group(1).lower()
    methods_raw = methods_match.group(1)

    # Sprytne dzielenie po przecinku i usuwanie spacji (naprawia błąd ze screena!)
    methods_list = [m.strip().lower() for m in methods_raw.split(',') if m.strip()]

    await update.message.reply_text(f"⏳ Rozpoczynam wielkie starcie algorytmów na zbiorze {dataset_name.upper()}!\nBiorące udział metody: {', '.join(methods_list)}\nTo potrwa od kilkunastu sekund do kilku minut...")

    try:
        # Puszczamy to w tle, żeby bot się nie zawiesił
        raport, plot_path = await asyncio.to_thread(dl_models.run_comparison, dataset_name, methods_list)
        await update.message.reply_text(raport, parse_mode="Markdown")
        if os.path.exists(plot_path):
            await update.message.reply_photo(photo=open(plot_path, 'rb'))
    except Exception as e:
        import traceback
        traceback.print_exc()
        await update.message.reply_text(f"❌ Błąd podczas porównania: {str(e)}")


# ==========================================
#      ZACHOWANE KOMENDY Z LAB 1 i 2
# ==========================================

async def classify_lab2_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.replace("/classify", "").strip()
    args_dict = {part.split("=", 1)[0].lower(): part.split("=", 1)[1].lower() for part in text.split() if "=" in part}
    
    required_keys = ["dataset", "method", "gridsearch", "run"]
    missing_keys = [k for k in required_keys if k not in args_dict]
    
    if missing_keys:
        await update.message.reply_text(f"Błąd! Brakuje parametrów: {', '.join(missing_keys)}.")
        return

    dataset, method = args_dict["dataset"], args_dict["method"]
    gridsearch = True if args_dict["gridsearch"] == "true" else False 
    
    try: run_count = int(args_dict["run"])
    except ValueError:
        await update.message.reply_text("Błąd: Parametr 'run' musi być liczbą.")
        return

    await update.message.reply_text(f"🚀 Rozpoczynam eksperyment (Lab 2)...")
    try:
        raport = await asyncio.to_thread(lab2_experiments.run_experiment, dataset, method, gridsearch, run_count)
        await update.message.reply_text(f"✅ Eksperyment zakończony!\n\n{raport}")
    except Exception as e:
        await update.message.reply_text(f"❌ Błąd podczas obliczeń: {e}")

async def task_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = parse_args(update.message.text)
    if len(args) != 3:
        await update.message.reply_text("Błąd składni. Użycie: /task <nazwa_zadania> \"tekst\" \"klasa\"")
        return

    task_name, text, text_class = args
    data_handler.save_record(text, text_class)
    
    tokens = nlp_tools.tokenize_text(text)
    response_text, photo_path = "", None

    if task_name == "tokenize": response_text = f"Tokeny: {tokens}"
    elif task_name == "remove_stopwords": response_text = f"Bez stop words: {nlp_tools.remove_stopwords(tokens)}"
    elif task_name == "lemmatize": response_text = f"Lematyzacja: {nlp_tools.lemmatize_tokens(tokens)}"
    elif task_name == "stemming": response_text = f"Stemming: {nlp_tools.stem_tokens(tokens)}"
    elif task_name == "n-grams": response_text = f"Bigramy: {nlp_tools.generate_ngrams(tokens, 2)}"
    elif task_name == "plot_histogram":
        photo_path = nlp_tools.plot_histogram(tokens)
        response_text = "Wygenerowano histogram długości tokenów."
    elif task_name == "plot_wordcloud":
        photo_path = nlp_tools.plot_wordcloud(text)
        response_text = "Wygenerowano word cloud."
    elif task_name == "stats": response_text = f"Liczba znaków: {len(text)}, tokenów: {len(tokens)}"
    else: response_text = "Nieznane zadanie."

    await update.message.reply_text(response_text)
    if photo_path and os.path.exists(photo_path):
        await update.message.reply_photo(photo=open(photo_path, 'rb'))

async def full_pipeline_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = parse_args(update.message.text)
    if len(args) != 2:
        await update.message.reply_text("Błąd składni. Użycie: /full_pipeline \"tekst\" \"klasa\"")
        return
    await update.message.reply_text("Uruchamianie pełnego pipeline'u (sprawdź konsolę)...")
    # Kod działa w tle na serwerze - ograniczyliśmy spam na czacie

async def classifier_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = parse_args(update.message.text)
    if len(args) != 1:
        await update.message.reply_text("Błąd składni. Użycie: /classifier \"tekst\"")
        return
    await update.message.reply_text(ml_model.train_and_predict(args[0]))

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = data_handler.load_data()
    if not data:
        await update.message.reply_text("Baza danych z Lab 1 jest pusta.")
        return
    await update.message.reply_text(f"Zbiór danych zawiera {len(data)} rekordów z Lab 1.")

# ==========================================
#         KOMENDY Z LAB 4 (NOWE)
# ==========================================

async def ner_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_msg = update.message.text
    import re
    method_match = re.search(r'method=(\w+)', text_msg)
    text_match = re.search(r'text="([^"]+)"', text_msg)

    if not method_match or not text_match:
        await update.message.reply_text("❌ Użycie: `/ner method=<spacy|stanza> text=\"Twój tekst\"`", parse_mode="Markdown")
        return

    method = method_match.group(1).lower()
    text = text_match.group(1)

    await update.message.reply_text(f"⏳ Rozpoczynam analizę NER metodą {method}...")
    result = await asyncio.to_thread(nlp_advanced.perform_ner, text, method)
    await update.message.reply_text(result)

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_msg = update.message.text
    import re
    target_match = re.search(r'target_lang=(\w+)', text_msg)
    text_match = re.search(r'text="([^"]+)"', text_msg)

    if not target_match or not text_match:
        await update.message.reply_text("❌ Użycie: `/translate text=\"tekst\" target_lang=<en|pl|de|fr>`", parse_mode="Markdown")
        return

    target_lang = target_match.group(1).lower()
    text = text_match.group(1)

    await update.message.reply_text("⏳ Uruchamiam lokalne modele tłumaczące (Hugging Face)...")
    translated_text, source_lang = await asyncio.to_thread(translation_tools.translate_text, text, target_lang)
    
    raport = (f"**Source:** {source_lang}\n**Target:** {target_lang}\n\n**Translation:**\n{translated_text}")
    await update.message.reply_text(raport, parse_mode="Markdown")

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_msg = update.message.text
    import re
    type_match = re.search(r'type=(\w+)', text_msg)
    len_match = re.search(r'length=(\w+)', text_msg)
    text_match = re.search(r'text="([^"]+)"', text_msg)

    if not type_match or not len_match or not text_match:
        await update.message.reply_text("❌ Użycie: `/summarize text=\"tekst\" type=<abstractive|extractive|bullets> length=<short|medium|long>`", parse_mode="Markdown")
        return

    s_type = type_match.group(1).lower()
    s_len = len_match.group(1).lower()
    text = text_match.group(1)

    await update.message.reply_text(f"⏳ Twój lokalny model LLM (Ollama) czyta tekst i przygotowuje podsumowanie...")
    result = await asyncio.to_thread(llm_tools.generate_summary, text, s_type, s_len)
    await update.message.reply_text(result)

async def analyze_entities_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_msg = update.message.text
    import re
    link_match = re.search(r'link=(\w+)', text_msg)
    text_match = re.search(r'text="([^"]+)"', text_msg)

    if not text_match:
        await update.message.reply_text("❌ Użycie: `/analyze_entities text=\"tekst\" link=<true|false>`", parse_mode="Markdown")
        return

    text = text_match.group(1)
    do_link = link_match.group(1).lower() == 'true' if link_match else False

    await update.message.reply_text("⏳ Szukam entitetów, linkuję z Wikipedią i rysuję graf...")
    
    # 1. Wyciągamy entitety (używamy domyślnie Spacy dla szybkości)
    # Wymaga sprytnego sparsowania wyniku ze stringa, który zwraca funkcja perform_ner
    ner_result = await asyncio.to_thread(nlp_advanced.perform_ner, text, "spacy")
    
    # Ręczne wyciągnięcie samych nazw z outputu (Dla grafu wiedzy)
    entities_for_graph = []
    lines = ner_result.split('\n')
    for line in lines:
        if line.startswith('- '):
            # Format: - Nazwa (TYP) [start:end]
            parts = line.split(' (')
            if len(parts) > 1:
                name = parts[0][2:].strip() # Usunięcie spacji na końcu!
                ent_type = parts[1].split(')')[0].strip()
                
                # Szybki filtr odrzucający halucynacje słabego modelu Spacy
                if name.lower() not in ["oraz", "i", "w", "z", "do"]:
                    entities_for_graph.append((name, ent_type))
    if not entities_for_graph:
        await update.message.reply_text("Nie znaleziono żadnych osób, miejsc ani organizacji do analizy.")
        return

    # 2. Generowanie raportu
    raport = "🔍 **ZNAJDŹ I POŁĄCZ (NEL):**\n\n"
    for name, e_type in entities_for_graph:
        raport += f"🔹 **{name}** ({e_type})\n"
        if do_link:
            wiki_data = await asyncio.to_thread(knowledge_graph.link_entity_wikipedia, name)
            if wiki_data:
                top_hit = wiki_data[0]
                raport += f"   🔗 Wikipedia: [{top_hit['title']}]({top_hit['url']})\n"
                raport += f"   📖 Opis: _{top_hit['summary']}_\n"
            else:
                raport += "   🔗 Wikipedia: Nie znaleziono\n"
        raport += "\n"

    await update.message.reply_text(raport, parse_mode="Markdown")

    # 3. Rysowanie i wysyłanie grafu
    plot_path = await asyncio.to_thread(knowledge_graph.generate_knowledge_graph, entities_for_graph, text)
    import os
    if os.path.exists(plot_path):
        await update.message.reply_photo(photo=open(plot_path, 'rb'), caption="Twój Graf Wiedzy 🕸️")

async def language_detect_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_msg = update.message.text
    import re
    text_match = re.search(r'text="([^"]+)"', text_msg)

    if not text_match:
        await update.message.reply_text("❌ Użycie: `/language_detect text=\"Twój tekst\"`", parse_mode="Markdown")
        return

    text = text_match.group(1)
    # Używamy funkcji, którą napisaliśmy wcześniej w nlp_advanced.py
    lang = await asyncio.to_thread(nlp_advanced.detect_lang, text)
    
    await update.message.reply_text(f"🌍 Wykryty język tekstu: **{lang.upper()}**", parse_mode="Markdown")

# ==========================================
#         KOMENDY Z LAB 5 (AGENT & VISION)
# ==========================================

async def agent_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługuje zapytania do inteligentnego agenta Ollama z Tool Callingiem."""
    text_msg = update.message.text
    # Usuwamy słowo '/agent' z początku wiadomości
    query = text_msg.replace('/agent', '').strip()

    if not query:
        await update.message.reply_text("❌ Użycie: `/agent <Twoje zapytanie>`\nNp. `/agent Jaka jest pogoda w Londynie?`", parse_mode="Markdown")
        return

    await update.message.reply_text("🤖 [Agent myśli...] Analizuję zapytanie i dobieram narzędzia...")
    
    # Wywołanie agenta (działa w tle, żeby nie blokować bota)
    result = await asyncio.to_thread(llm_agent.chat_with_agent, query)
    await update.message.reply_text(result)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Odbiera zdjęcie z Telegrama i wysyła do modelu Vision."""
    await update.message.reply_text("👁️ Otrzymałem zdjęcie. Pobieram i uruchamiam analizę wizyjną...")
    
    # Pobieramy zdjęcie w najwyższej jakości (ostatni element z listy)
    photo_file = await update.message.photo[-1].get_file()
    
    # Tworzymy tymczasowy folder na zdjęcia
    os.makedirs("temp_images", exist_ok=True)
    file_path = os.path.join("temp_images", f"{photo_file.file_id}.jpg")
    
    # Zapisujemy na dysk
    await photo_file.download_to_drive(file_path)
    
    # Użytkownik może dodać podpis do zdjęcia jako pytanie (prompt)
    prompt = update.message.caption if update.message.caption else "Opisz dokładnie, co widzisz na tym zdjęciu."
    
    # Wysyłamy do Ollamy
    result = await asyncio.to_thread(vision_tools.analyze_image, file_path, prompt)
    
    # Odpowiadamy i sprzątamy plik z dysku
    await update.message.reply_text(f"🖼️ **Analiza obrazu:**\n{result}", parse_mode="Markdown")
    if os.path.exists(file_path):
        os.remove(file_path)