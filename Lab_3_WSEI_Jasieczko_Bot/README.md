# Telegram NLP & ML Bot - Laboratorium 1, 2 i 3

Zaawansowany bot na platformę Telegram do przetwarzania języka naturalnego (NLP), przeprowadzania eksperymentów z zakresu uczenia maszynowego (Machine Learning) na pełnych zbiorach danych oraz trenowania modeli Głębokiego Uczenia (Deep Learning) do analizy sentymentu.

## Wymagania
- Python 3.12.7
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
   `pip install python-telegram-bot nltk scikit-learn matplotlib seaborn wordcloud python-dotenv gensim pandas datasets matplotlib seaborn wordcloud tensorflow transformers textblob stanza tf-keras`
3. Utwórz nowy plik o nazwie .env w głównym folderze projektu. "TELEGRAM_TOKEN=Twój_Token_Od_BotFathera"


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