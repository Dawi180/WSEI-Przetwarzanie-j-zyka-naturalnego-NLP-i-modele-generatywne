import os
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim.downloader as api
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Pobranie tokenizatora do W2V/GloVe
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

PLOTS_DIR = 'lab2plots'
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

RESULTS_FILE = 'lab2results.csv'

# --- 1. WCZYTYWANIE DANYCH ---
def load_dataset_data(dataset_name):
    print(f"Pobieranie datasetu: {dataset_name}...")
    if dataset_name == "20news_group":
        cats = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
        data = fetch_20newsgroups(subset='all', categories=cats, remove=('headers', 'footers', 'quotes'))
        return list(data.data[:2000]), list(data.target[:2000]), list(data.target_names)
    elif dataset_name == "imdb":
        dataset = load_dataset("imdb")
        train = dataset['train'].shuffle(seed=42).select(range(1500))
        test = dataset['test'].shuffle(seed=42).select(range(500))
        return list(train['text']) + list(test['text']), list(train['label']) + list(test['label']), ["negative", "positive"]
    elif dataset_name == "ag_news":
        dataset = load_dataset("ag_news")
        train = dataset['train'].shuffle(seed=42).select(range(1500))
        test = dataset['test'].shuffle(seed=42).select(range(500))
        return list(train['text']) + list(test['text']), list(train['label']) + list(test['label']), ["World", "Sports", "Business", "Sci/Tech"]
    elif dataset_name == "amazon":
        dataset = load_dataset("amazon_polarity")
        train = dataset['train'].shuffle(seed=42).select(range(1500))
        test = dataset['test'].shuffle(seed=42).select(range(500))
        return list(train['content']) + list(test['content']), list(train['label']) + list(test['label']), ["negative", "positive"]
    else:
        raise ValueError(f"Nieznany dataset: {dataset_name}")

# --- 2. REPREZENTACJE TEKSTU ---
def get_document_embedding(tokens, model, vector_size, is_gensim_api=False):
    if is_gensim_api: valid_words = [word for word in tokens if word in model]
    else: valid_words = [word for word in tokens if word in model.wv]
        
    if not valid_words: return np.zeros(vector_size)
    if is_gensim_api: return np.mean([model[word] for word in valid_words], axis=0)
    else: return np.mean([model.wv[word] for word in valid_words], axis=0)

def prepare_representations(texts):
    print("Generowanie reprezentacji...")
    representations = {}
    
    # Przechowujemy też oryginalne wektoryzatory do Feature Importance
    tfidf_vec = TfidfVectorizer(max_features=1000)
    representations['tfidf'] = tfidf_vec.fit_transform(texts).toarray()
    representations['tfidf_vectorizer'] = tfidf_vec
    
    bow_vec = CountVectorizer(max_features=1000)
    representations['bow'] = bow_vec.fit_transform(texts).toarray()
    
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4)
    representations['word2vec'] = np.array([get_document_embedding(tokens, w2v_model, 100, False) for tokens in tokenized_texts])
    representations['word2vec_model'] = w2v_model
    
    glove_model = api.load("glove-wiki-gigaword-50")
    representations['glove'] = np.array([get_document_embedding(tokens, glove_model, 50, True) for tokens in tokenized_texts])
    return representations

# --- 3. WIZUALIZACJE I EKSPERTY (NOWOŚĆ) ---

def generate_wordclouds(texts, labels, target_names):
    print("Generowanie Word Cloud...")
    # Corpus
    wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(texts))
    wc.to_file(f"{PLOTS_DIR}/wordcloud_corpus.png")
    
    # Dla każdej klasy
    for idx, class_name in enumerate(target_names):
        class_texts = [texts[i] for i in range(len(texts)) if labels[i] == idx]
        if class_texts:
            wc_class = WordCloud(width=800, height=400, background_color='white').generate(" ".join(class_texts))
            # Bezpieczna nazwa pliku (usuwamy np. ukośniki z Sci/Tech)
            safe_name = class_name.replace("/", "_").replace("\\", "_")
            wc_class.to_file(f"{PLOTS_DIR}/wordcloud_class_{safe_name}.png")

def plot_confusion_matrix(y_true, y_pred, emb_name, mod_name, target_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title(f'Confusion Matrix: {emb_name} + {mod_name}')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/confusion_{emb_name}_{mod_name}.png")
    plt.close()

def plot_document_embeddings(X, labels, dataset_name, mod_name, emb_name):
    # Skracamy próbkę do 500 żeby t-SNE nie liczyło się wiekami
    sample_size = min(500, len(X))
    X_sample, labels_sample = X[:sample_size], labels[:sample_size]
    
    reductions = {
        'pca': PCA(n_components=2),
        'tsne': TSNE(n_components=2, perplexity=30, random_state=42),
        'svd': TruncatedSVD(n_components=2)
    }
    
    for red_name, reducer in reductions.items():
        try:
            X_red = reducer.fit_transform(X_sample)
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X_red[:, 0], X_red[:, 1], c=labels_sample, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Klasa')
            plt.title(f'{red_name.upper()} - {dataset_name} ({emb_name})')
            plt.savefig(f"{PLOTS_DIR}/{dataset_name}_{mod_name}_{emb_name}_{red_name}_embedding.png")
            plt.close()
        except Exception as e:
            print(f"Błąd przy redukcji {red_name}: {e}")

def save_similar_words(w2v_model):
    print("Zapisywanie podobnych słów...")
    words_to_check = ['space', 'computer', 'science', 'music', 'car']
    with open('lab2_similar_words.txt', 'w', encoding='utf-8') as f:
        f.write("Podobne słowa z wyuczonego Word2Vec:\n")
        for w in words_to_check:
            if w in w2v_model.wv:
                sims = w2v_model.wv.most_similar(w, topn=5)
                f.write(f"{w}: {sims}\n")
            else:
                f.write(f"{w}: [Brak słowa w słowniku tego korpusu]\n")

def plot_word_embeddings(w2v_model):
    print("Wizualizacja słów w embeddingu...")
    # Bierzemy 100 najpopularniejszych słów
    words = list(w2v_model.wv.index_to_key)[:100]
    if not words: return
    vectors = np.array([w2v_model.wv[w] for w in words])
    
    reductions = {'pca': PCA(n_components=2), 'tsne': TSNE(n_components=2, perplexity=10, random_state=42)}
    for red_name, reducer in reductions.items():
        vecs_red = reducer.fit_transform(vectors)
        plt.figure(figsize=(12, 10))
        plt.scatter(vecs_red[:, 0], vecs_red[:, 1], color='red')
        for i, word in enumerate(words):
            plt.annotate(word, xy=(vecs_red[i, 0], vecs_red[i, 1]), fontsize=9)
        plt.title(f'Word Embeddings - {red_name.upper()}')
        plt.savefig(f"{PLOTS_DIR}/word_embedding_{red_name}.png")
        plt.close()

def save_feature_importance(model, vectorizer, target_names):
    # Wyciąga Feature Importance z Logistic Regression
    if hasattr(model, 'coef_'):
        importance = model.coef_[0]
        feature_names = vectorizer.get_feature_names_out()
        top10_idx = np.argsort(importance)[-10:]
        
        with open('lab2_feature_importance.txt', 'w', encoding='utf-8') as f:
            f.write("Top 10 najważniejszych cech (TF-IDF + LogReg) dla pierwszej klasy:\n")
            for i in reversed(top10_idx):
                f.write(f"{feature_names[i]}: {importance[i]:.4f}\n")

# --- 4. GŁÓWNA FUNKCJA BOTA ---
def get_models():
    return {'nb': MultinomialNB(), 'rf': RandomForestClassifier(random_state=42), 
            'mlp': MLPClassifier(max_iter=500, random_state=42), 'logreg': LogisticRegression(max_iter=1000, random_state=42)}

def get_grids():
    return {'nb': {'alpha': [0.1, 1.0]}, 'rf': {'n_estimators': [100]}, 
            'logreg': {'C': [0.1, 1]}, 'mlp': {'hidden_layer_sizes': [(128,)]}}

def create_or_update_csv(embedding, model, accuracy, macro_f1, seed):
    file_exists = os.path.isfile(RESULTS_FILE)
    df = pd.DataFrame({"embedding": [embedding], "model": [model], "accuracy": [accuracy], "macro_f1": [macro_f1], "seed": [seed]})
    df.to_csv(RESULTS_FILE, mode='a', header=not file_exists, index=False)

def run_experiment(dataset_name, method, gridsearch, run_count):
    try:
        texts, labels, target_names = load_dataset_data(dataset_name)
        reps = prepare_representations(texts)
        
        # Generowanie podstawowych wykresów NLP
        generate_wordclouds(texts, labels, target_names)
        save_similar_words(reps['word2vec_model'])
        plot_word_embeddings(reps['word2vec_model'])
        
        seeds = [42, 1337, 2024][:run_count]
        models_to_run = ['nb', 'rf', 'mlp', 'logreg'] if method == 'all' else method.split(',')
        results_summary = []
        
        for seed in seeds:
            print(f"\n--- Uruchomienie dla seed: {seed} ---")
            for emb_name, X in reps.items():
                if emb_name in ['word2vec_model', 'glove_model', 'tfidf_vectorizer']: continue
                
                X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=seed)
                
                for mod_name in models_to_run:
                    if mod_name not in get_models(): continue
                    print(f"Trenowanie i wizualizacja: {emb_name} + {mod_name}")
                    
                    model = get_models()[mod_name]
                    if mod_name == 'nb' and emb_name in ['word2vec', 'glove']:
                        scaler = MinMaxScaler()
                        X_train_model = scaler.fit_transform(X_train)
                        X_test_model = scaler.transform(X_test)
                    else:
                        X_train_model, X_test_model = X_train, X_test
                        
                    clf = GridSearchCV(model, get_grids()[mod_name], cv=2, n_jobs=-1) if gridsearch else model
                    clf.fit(X_train_model, y_train)
                    y_pred = clf.predict(X_test_model)
                    
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='macro')
                    create_or_update_csv(emb_name, mod_name, acc, f1, seed)
                    results_summary.append(f"🌱 Seed {seed} | {emb_name} + {mod_name} | Acc: {acc:.2f}")
                    
                    # Wizualizacje wymagane w instrukcji
                    best_estimator = clf.best_estimator_ if gridsearch else clf
                    plot_confusion_matrix(y_test, y_pred, emb_name, mod_name, target_names)
                    plot_document_embeddings(X_test_model, y_test, dataset_name, mod_name, emb_name)
                    
                    if mod_name == 'logreg' and emb_name == 'tfidf':
                        save_feature_importance(best_estimator, reps['tfidf_vectorizer'], target_names)

        return f"Eksperyment ukończony pełnym sukcesem! Wygenerowano mnóstwo wykresów w folderze {PLOTS_DIR} oraz pliki tekstowe.\n\nPodsumowanie wyników:\n" + "\n".join(results_summary)
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Błąd w eksperymencie: {str(e)}"