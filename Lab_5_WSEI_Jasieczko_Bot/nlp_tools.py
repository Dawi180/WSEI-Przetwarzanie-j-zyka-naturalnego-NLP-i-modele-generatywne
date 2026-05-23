import os
import re
import urllib.request
from datetime import datetime
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.util import ngrams

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Pobieranie niezbędnych danych NLTK (wykona się tylko raz)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True) # Wymagane w nowszych wersjach NLTK

# Upewniamy się, że folder na wykresy istnieje
PLOTS_DIR = 'plots'
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def get_plot_filename(plot_type="plot"):
    """Generuje nazwę pliku w wymaganym formacie, z dodatkiem typu wykresu."""
    now = datetime.now().strftime("%Y-%m-%d}_{%H-%M-%S")
    now = now.replace("}_{", "_")
    # Dodajemy typ wykresu na koniec nazwy, aby pliki się nie nadpisywały w tej samej sekundzie
    return os.path.join(PLOTS_DIR, f"Sentence_{now}_{plot_type}.png")

def clean_text(text):
    """Usuwa znaki interpunkcyjne i zamienia na małe litery."""
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def tokenize_text(text):
    """Dzieli tekst na tokeny (słowa)."""
    return word_tokenize(text)

def split_sentences(text):
    """Dzieli tekst na zdania."""
    return sent_tokenize(text)

STOPWORDS_FILE = 'stopwords-pl.txt'
STOPWORDS_URL = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-pl/master/stopwords-pl.txt'

def get_polish_stopwords():
    """Pobiera i wczytuje polskie stop words z GitHuba, jeśli jeszcze ich nie ma."""
    if not os.path.exists(STOPWORDS_FILE):
        print("Pobieranie polskich stopwords z GitHuba...")
        try:
            urllib.request.urlretrieve(STOPWORDS_URL, STOPWORDS_FILE)
        except Exception as e:
            print(f"Błąd podczas pobierania stopwords: {e}")
            return set() # W razie błędu braku internetu zwraca pusty zbiór
            
    # Wczytanie słów z pobranego pliku
    with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f if line.strip())
    return stopwords

def remove_stopwords(tokens):
    """Usuwa polskie stop words na podstawie dedykowanej listy."""
    stop_words = get_polish_stopwords()
    return [word for word in tokens if word.lower() not in stop_words]

def lemmatize_tokens(tokens):
    """Lematyzacja (sprowadzanie do formy podstawowej)."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def stem_tokens(tokens):
    """Stemming (obcinanie końcówek)."""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def generate_ngrams(tokens, n=2):
    """Generuje n-gramy."""
    return list(ngrams(tokens, n))

def get_bow(text_list):
    """Zwraca reprezentację Bag of Words."""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text_list)
    return vectorizer.vocabulary_, X.toarray()

def get_tfidf(text_list):
    """Zwraca reprezentację TF-IDF."""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_list)
    return vectorizer.vocabulary_, X.toarray()

# --- WIZUALIZACJE ---

def plot_histogram(tokens):
    """Rysuje i zapisuje histogram długości tokenów."""
    lengths = [len(token) for token in tokens]
    plt.figure(figsize=(8, 6))
    plt.hist(lengths, bins=range(1, max(lengths) + 2 if lengths else 2), align='left', color='skyblue', edgecolor='black')
    plt.title('Histogram długości tokenów')
    plt.xlabel('Długość tokenu')
    plt.ylabel('Częstość')
    
    filepath = get_plot_filename("hist")
    plt.savefig(filepath)
    plt.close()
    return filepath

def plot_wordcloud(text):
    """Generuje i zapisuje chmurę słów."""
    plt.figure(figsize=(8, 6))
    if not text.strip():
        text = "Brak słów" # Zabezpieczenie przed pustym tekstem
        
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    filepath = get_plot_filename("wc")
    plt.savefig(filepath)
    plt.close()
    return filepath

def plot_bar_chart(tokens):
    """Rysuje wykres słupkowy najczęstszych słów."""
    counts = Counter(tokens)
    common = counts.most_common(10) # 10 najczęstszych
    
    if not common:
        return None

    words, frequencies = zip(*common)
    
    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies, color='lightgreen')
    plt.title('Najczęstsze słowa')
    plt.xlabel('Słowa')
    plt.ylabel('Wystąpienia')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filepath = get_plot_filename("bar")
    plt.savefig(filepath)
    plt.close()
    return filepath