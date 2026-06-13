# Telegram NLP & ML Bot - Laboratorium 1-5

Zaawansowany bot na platformę Telegram do przetwarzania języka naturalnego (NLP), uczenia maszynowego (ML), Głębokiego Uczenia (DL) oraz wykorzystania lokalnych modeli generatywnych (LLM - Ollama), tłumaczeń offline, budowy Grafów Wiedzy oraz inteligentnych Agentów AI (Tool Calling & Vision).

## Wymagania
- Python 3.12.7
- Utworzone wirtualne środowisko (venv)
- Windows:
     ```bash
     py -3.12 -m venv venv
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
   `pip install python-telegram-bot nltk scikit-learn matplotlib seaborn wordcloud python-dotenv gensim pandas datasets matplotlib seaborn wordcloud tensorflow transformers textblob stanza tf-keras spacy networkx langdetect wikipedia requests torch sentencepiece sacremoses beautifulsoup4 duckduckgo-search ollama`
3. Pobierz modele językowe dla biblioteki Spacy (wymagane do NER):
    ```bash
    python -m spacy download en_core_web_sm
    python -m spacy download pl_core_news_sm
    ```
4. Zainstaluj lokalne środowisko LLM (Ollama):
    Pobierz i zainstaluj program Ollama ze strony: ollama.com
    Po instalacji otwórz nowy terminal i pobierz model (np. lekki model phi3), wpisując:
    ```bash
    ollama pull phi3       # (Lab 4 - Streszczenia)
    ollama pull qwen2.5    # (Lab 5 - Agent & Tool Calling)
    ollama pull minicpm-v  # (Lab 5 - Moduł Vision)
    ```
    (Uwaga: Aplikacja Ollama musi być uruchomiona w tle na komputerze, aby bot mógł generować streszczenia).
5. Utwórz nowy plik o nazwie .env w głównym folderze projektu. "TELEGRAM_TOKEN=Twój_Token_Od_BotFathera"


## Uruchomienie
Wpisz w terminalu:
`python bot.py`

## Dostępne komendy i przykłady użycia
### 🛠️ Część 1: Analiza pojedynczych wiadomości (Lab 1)
Bot buduje na bieżąco lokalną bazę danych w pliku `sentences.json` i pozwala na analizę pojedynczych zdań.

* **`/task <nazwa_zadania> "tekst" "klasa"`** - Wykonuje wybrane zadanie NLP na tekście.
  * *Dostępne zadania:* `tokenize`, `remove_stopwords`, `lemmatize`, `stemming`, `n-grams`, `plot_histogram`, `plot_wordcloud`.
  * *Przykład:* `/task tokenize "To jest testowy tekst bota." "neutralny"`
  * *Przykład:* `/task plot_wordcloud "Bardzo lubię sztuczną inteligencję." "pozytywny"`
* **`/full_pipeline "tekst" "klasa"`** - Przechodzi przez cały proces NLP, generuje statystyki i wykresy.
  * *Przykład:* `/full_pipeline "Ten system działa strasznie wolno i zacina się." "negatywny"`
* **`/classifier "tekst"`** - Klasyfikuje tekst na podstawie danych zebranych wcześniej za pomocą komend powyżej.
  * *Przykład:* `/classifier "Bardzo fajny film"`
* **`/stats`** - Generuje statystyki, Word Cloud, histogramy oraz liczność n-gramów dla całego zebranego zbioru z pliku JSON.
  * *Przykład:* `/stats`

---

### 🚀 Część 2: Eksperymenty na całych datasetach (Lab 2)
Bot pozwala na pobieranie dużych zbiorów danych, generowanie wektorów (BoW, TF-IDF, Word2Vec, GloVe), trenowanie modeli ML (Naive Bayes, Random Forest, Logistic Regression, MLP) oraz tworzenie zaawansowanych wizualizacji (PCA, t-SNE, SVD).

* **`/classify dataset=<dataset> method=<model> gridsearch=<true/false> run=<n>`**
  * *Dostępne datasety:* `20news_group`, `imdb`, `ag_news`, `amazon`
  * *Dostępne metody (modele):* `nb`, `rf`, `logreg`, `mlp`, `all`
  
  **Przykłady użycia:**
  * *Szybki test jednego modelu bez GridSearch:* `/classify dataset=20news_group method=nb gridsearch=false run=1`
  * *Porównanie wszystkich modeli (uwaga: może zająć dużo czasu!):* `/classify dataset=imdb method=all gridsearch=false run=1`
  * *Uruchomienie strojenia hiperparametrów (GridSearch) dla Regresji Logistycznej:* `/classify dataset=ag_news method=logreg gridsearch=true run=2`

  ### 🧠 Część 3: Deep Learning i Analiza Sentymentu (Lab 3)
Bot został przebudowany na architekturę modułową. Zyskał potężne algorytmy do analizy sentymentu, w tym modele sekwencyjne z użyciem TensorFlow (RNN, LSTM, GRU) oraz obsługę Transformerów i biblioteki Stanza.

* **`/help`** - Wyświetla krótką pomoc i listę dostępnych komend bezpośrednio na czacie.
* **`/add_sentiment "tekst" "etykieta"`** - Dopisuje nowy rekord do własnego, lokalnego zbioru danych (`sentiment_dataset.csv`).
  * *Przykład:* `/add_sentiment "Ten produkt był absolutnie fantastyczny!" "pozytywny"`
* **`/sentiment method=<metoda> text="tekst"`** - Analizuje sentyment podanego tekstu w locie. Wykorzystuje wczytane z dysku modele `.h5` (jeśli wybrano sieć) lub gotowe rozwiązania NLP.
  * *Dostępne metody:* `rule`, `textblob`, `transformer`, `stanza`, `simplernn`, `lstm`, `gru`
  * *Przykład:* `/sentiment method=transformer text="This movie was terrible and boring."`
* **`/train model=<model> dataset=<dataset>`** - Uruchamia proces uczenia sieci neuronowej na danym zbiorze danych. Zapisuje wagi modelu i tokenizer w folderze `models/` oraz rysuje wykres historii uczenia (Loss/Accuracy).
  * *Dostępne modele:* `simplernn`, `lstm`, `gru`
  * *Dostępne datasety:* `amazon`, `imdb`, `custom`
  * *Przykład:* `/train model=lstm dataset=imdb`
* **`/compare dataset=<dataset> methods=<lista_metod>`** - Rozpoczyna wielkie starcie wybranych metod na podanym zbiorze. Generuje statystyki (Accuracy, F1, Precision, Recall), zapisuje je do pliku CSV i odsyła wygenerowany wykres słupkowy.
  * *Przykład:* `/compare dataset=imdb methods=rule, textblob, transformer, lstm`
* **`/models`** - Wyświetla listę wszystkich wytrenowanych dotychczas modeli gotowych do użycia (plików `.h5`).

  ### 🤖 Część 4: Zaawansowane NLP, LLM i Grafy Wiedzy (Lab 4)
  Kompleksowy bot do przetwarzania języka naturalnego (NLP) wykorzystujący lokalne modele AI, tłumaczenia offline oraz integrację z Wikipedią. Projekt skupia się na prywatności danych i wydajności poprzez uruchamianie modeli bezpośrednio na lokalnej maszynie.


## 🚀 Główne Funkcjonalności

* **NER (Named Entity Recognition):** Rozpoznawanie osób, organizacji i lokalizacji przy użyciu bibliotek spaCy oraz Stanza.
* **NEL (Named Entity Linking):** Łączenie wykrytych bytów z bazą wiedzy Wikipedii.
* **Knowledge Graphs:** Automatyczne generowanie wizualnych grafów powiązań (PNG).
* **Offline Translation:** Tłumaczenie tekstów bez dostępu do chmury (MarianMT).
* **LLM Summarization:** Inteligentne streszczenia tekstów dzięki integracji z **Ollama**.

  ### Część 5: Inteligentny Agent i Analiza Obrazu (Lab 5)
  * /agent <zapytanie> - Uruchamia autonomicznego agenta LLM (Llama 3.2) wyposażonego w system narzędzi (Function Calling).
  * Dostępne narzędzia: Kalkulator matematyczny, Lokalna Baza Wiedzy, Wyszukiwarka Internetowa (DuckDuckGo), API Pogodowe (Open-Meteo z geokodowaniem).
  * Przykład: `/agent Jaka jest dzisiaj pogoda w Krakowie i ile to jest pierwiastek z 144 pomnożony przez 5?
  * Analiza Obrazu (Vision) - Bot automatycznie reaguje na przesłane pliki graficzne/zdjęcia. Jeśli dodasz podpis (np. "Opisz to zdjęcie"), lokalny model multimodalny minicpm-v dokona pełnej analizy wizualnej i odeśle tekstowy opis.

---

## 🛠 Komendy i Użycie

### 🔍 Rozpoznawanie encji (NER)
Rozpoznaje nazwane byty w języku polskim lub angielskim.
* **Komenda:** `/ner method=<spacy|stanza> text="tekst"`
* **Przykład:** `/ner method=spacy text="Steve Jobs, założyciel Apple'a, urodził się w San Francisco."`

### 🔗 Analiza i Graf Wiedzy (NEL)
Wyodrębnia byty, linkuje je do Wikipedii i generuje plik graficzny z grafem powiązań.
* **Komenda:** `/analyze_entities text="tekst" link=<true|false>`
* **Przykład:** `/analyze_entities text="Elon Musk posiada firmę Tesla oraz xAI w Austin." link=true`

### 🌍 Tłumaczenie Offline
Tłumaczenie przy użyciu natywnych modeli Hugging Face. W przypadku języków innych niż angielski, stosowany jest mechanizm "przesiadki" (pivot) przez język angielski.
* **Komenda:** `/translate text="tekst" target_lang=<en|pl|de|fr>`
* **Przykład:** `/translate text="Wczoraj kupiłem nowy samochód." target_lang=en`

### 📝 Podsumowanie (LLM)
Generuje streszczenie tekstu przy pomocy lokalnego modelu Ollama.
* **Komenda:** `/summarize text="tekst" type=<abstractive|extractive|bullets> length=<short|medium|long>`
* **Przykład:** `/summarize text="[Długi tekst...]" type=bullets length=short`

### 🌐 Detekcja Języka
Automatycznie rozpoznaje ponad 50 języków.
* **Komenda:** `/language_detect text="tekst"`
* **Przykład:** `/language_detect text="Guten Morgen! Ich lerne gerne neue Programmiersprachen."`

### 🛡️ Część 6: Automatyczna Moderacja i Bezpieczeństwo AI (Lab 6)
  Kompleksowy system wielopoziomowej moderacji treści (Content Moderation & Policy Enforcement) działający w architekturze hybrydowej (lokalne modele + API). System chroni społeczność przed wyciekiem danych wrażliwych, mową nienawiści oraz toksycznością za pomocą strategii głosowania modeli (Ensemble Strategy) oraz technologii Function Calling.

## 🚀 Główne Funkcjonalności

* **OpenAI Privacy Filter:** Automatyczna detekcja i usuwanie danych osobowych (PII) takich jak numery telefonów, adresy e-mail czy numery PESEL przed dalszym przetwarzaniem.
* **Bielik Guard 0.1B:** Lekki, dedykowany model bezpieczeństwa klasyfikujący zagrożenia (toxic, spam, hate_speech, self_harm, violence, sexual, clean) w języku polskim.
* **Ensemble Strategy & Function Calling:** Koordynacja potoków przez model **Qwen 2.5**, który na podstawie promptów systemowych i wyników modeli pomocniczych samodzielnie decyduje o wykonaniu akcji (zatwierdzenie, odrzucenie, ban, flagowanie).
* **Pętla Informacji Zwrotnej (Feedback Loop):** System zbiera korekty decyzji wprowadzane przez ludzkich moderatorów, buduje bazę treningową i pozwala na symulację re-rankingu decyzji (LLMOps).
* **Rejestr Recydywistów (Repeat Offenders):** Bot prowadzi stałą kartotekę punktów karnych i automatycznie nakłada blokady na użytkowników notorycznie łamiących regulamin.

---

## 🛠 Komendy i Użycie

### 📝 Moderacja Treści
Analizuje podany tekst przez pełną rurę przetwarzania (PII, Bielik Guard, Sentyment z Lab 3, NER z Lab 4) i podejmuje autonomiczną decyzję moderacyjną.
* **Komenda:** `/moderate <tekst do sprawdzenia>`
* **Przykład:** `/moderate Ty idioto, nienawidzę was! Mój pesel to 99010112345`

### 🔍 Status Zgłoszenia
Sprawdza szczegółowe wyniki archiwalnej moderacji i powody decyzji na podstawie unikalnego identyfikatora treści (Content ID).
* **Komenda:** `/mod_status <content_id>`
* **Przykład:** `/mod_status MSG-F69D4408`

### 📜 Historia Użytkownika (Kartoteka)
Wyciąga z bazy profil konkretnego użytkownika, pokazując łączną liczbę naruszeń, nałożone kary oraz status recydywisty.
* **Komenda:** `/mod_history <user_id>`
* **Przykład:** `/mod_history 123456789`

### 📊 Analityka Biznesowa
Generuje dynamiczny raport podsumowujący działania moderacyjne dla administratorów (procent odrzuceń, statystyki najczęstszych kategorii hejtu).
* **Komenda:** `/mod_analytics`
* **Przykład:** `/mod_analytics`

### ✍️ Korekta Ludzka (Feedback)
Pozwala administratorowi nadpisać błędną decyzję bota, zapisując próbkę uczącą do późniejszego dostrojenia systemu.
* **Komenda:** `/mod_add_feedback <content_id> <APPROVE|REJECT> <komentarz>`
* **Przykład:** `/mod_add_feedback MSG-F69D4408 APPROVE To legalna wypowiedź klienta, proszę przywrócić.`

### 👀 Lista Obserwowanych (Watchlist)
Wyświetla zestawienie wszystkich użytkowników, którzy mają na swoim koncie zarejestrowane naruszenia regulaminu.
* **Komenda:** `/mod_watchlist`
* **Przykład:** `/mod_watchlist`

### 🧠 Adaptacja i Re-ranking
Uruchamia pętlę ciągłego uczenia (Continuous Learning), dostosowując progi czułości Qwena na podstawie zebranych poprawek ludzkich.
* **Komenda:** `/mod_train_on_feedback`
* **Przykład:** `/mod_train_on_feedback`

### ⚖️ Sprawdzenie z Polityką Platformy
Szybka weryfikacja zgodności surowego tekstu z wewnętrznymi regułami bezpieczeństwa przed oficjalną publikacją.
* **Komenda:** `/mod_policy_check "<tekst>"`
* **Przykład:** `/mod_policy_check "Kupiłem produkt na stronie http://test.pl i nie działa"`

### ❓ Pomoc Moderacyjna
Wyświetla sformatowane menu podręczne ze spisem wszystkich komend administratorskich systemu bezpieczeństwa.
* **Komenda:** `/mod_help`
* **Przykład:** `/mod_help`