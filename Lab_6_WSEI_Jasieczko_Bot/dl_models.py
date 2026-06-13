import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from datasets import load_dataset
from config import MODELS_DIR, PLOTS_DIR_LAB3, DATASET_FILE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sentiment_methods

# Parametry globalne dla sieci (możesz z nimi eksperymentować)
MAX_WORDS = 5000
MAX_LEN = 100
EMBEDDING_DIM = 100
EPOCHS = 5

def load_sentiment_data(dataset_name):
    """Pobiera i formatuje dane z wybranego źródła."""
    if dataset_name == 'custom':
        if not os.path.exists(DATASET_FILE):
            raise FileNotFoundError(f"Brak pliku {DATASET_FILE}. Użyj najpierw /add_sentiment.")
        df = pd.read_csv(DATASET_FILE)
        if len(df) < 5:
            raise ValueError("Twój dataset jest za mały do treningu. Dodaj przynajmniej kilkanaście zdań!")
        return df['text'].astype(str).tolist(), df['label'].astype(str).tolist()
        
    elif dataset_name in ['imdb', 'amazon']:
        ds_name = "imdb" if dataset_name == 'imdb' else "amazon_polarity"
        text_col = "text" if dataset_name == 'imdb' else "content"
        ds = load_dataset(ds_name)
        # Próbkujemy dane (np. 1000 przykładów), żeby procesor nie liczył tego 3 godziny
        train = ds['train'].shuffle(seed=42).select(range(1000))
        test = ds['test'].shuffle(seed=42).select(range(200))
        
        # POPRAWKA: Rzutujemy obiekty Column na listy przed połączeniem
        texts = list(train[text_col]) + list(test[text_col])
        combined_labels = list(train['label']) + list(test['label'])
        
        # Mapowanie labelek: 1 -> pozytywny, 0 -> negatywny
        labels = ["pozytywny" if l == 1 else "negatywny" for l in combined_labels]
        return texts, labels
    else:
        raise ValueError(f"Nieznany dataset: {dataset_name}")

def train_and_save_model(model_type, dataset_name):
    """Buduje, trenuje i zapisuje wybrany model sekwencyjny."""
    print(f"Pobieranie danych: {dataset_name}...")
    texts, labels = load_sentiment_data(dataset_name)

    # 1. Kodowanie Etykiet
    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_classes = len(le.classes_)

    # 2. Tokenizacja i padding
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=MAX_LEN)

    # 3. Budowa Modelu (Embedding -> RNN/LSTM/GRU -> Dense)
    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN))

    if model_type == 'simplernn':
        model.add(SimpleRNN(64))
    elif model_type == 'lstm':
        model.add(LSTM(64))
    elif model_type == 'gru':
        model.add(GRU(64))
    else:
        raise ValueError("Zły typ modelu. Dostępne: simplernn, lstm, gru")

    model.add(Dense(32, activation='relu'))

    # Wybór funkcji strat w zależności od liczby klas (np. 2 w IMDB, 3 w Custom)
    if num_classes > 2:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 4. Trenowanie
    print(f"Rozpoczęto trening {model_type.upper()} na danych {dataset_name}...")
    history = model.fit(X, y, epochs=EPOCHS, batch_size=32, validation_split=0.2, verbose=1)

    # 5. Zapisywanie modeli (Zgodnie z wymaganiami Lab 3)
    model_path = os.path.join(MODELS_DIR, f"{model_type}_{dataset_name}.h5")
    tok_path = os.path.join(MODELS_DIR, f"{model_type}_{dataset_name}_tokenizer.pkl")
    le_path = os.path.join(MODELS_DIR, f"{model_type}_{dataset_name}_label_encoder.pkl")

    model.save(model_path)
    with open(tok_path, 'wb') as f: pickle.dump(tokenizer, f)
    with open(le_path, 'wb') as f: pickle.dump(le, f)

    # 6. Generowanie wykresu treningu
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Historia uczenia: {model_type.upper()} ({dataset_name})')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()
    plot_path = os.path.join(PLOTS_DIR_LAB3, f"train_history_{model_type}_{dataset_name}.png")
    plt.savefig(plot_path)
    plt.close()

    final_acc = history.history['val_accuracy'][-1]
    return model_path, plot_path, final_acc

def get_available_models():
    """Zwraca listę zapisanych modeli w folderze models/"""
    if not os.path.exists(MODELS_DIR): return []
    return [f for f in os.listdir(MODELS_DIR) if f.endswith('.h5')]

from tensorflow.keras.models import load_model

def predict_sentiment(text, model_type, dataset_name='imdb'):
    """Wczytuje zapisany model i przewiduje sentyment dla nowego tekstu."""
    model_path = os.path.join(MODELS_DIR, f"{model_type}_{dataset_name}.h5")
    tok_path = os.path.join(MODELS_DIR, f"{model_type}_{dataset_name}_tokenizer.pkl")
    le_path = os.path.join(MODELS_DIR, f"{model_type}_{dataset_name}_label_encoder.pkl")

    if not os.path.exists(model_path):
        return f"Błąd: Model {model_type} dla {dataset_name} nie istnieje. Wytrenuj go komendą /train!"

    # 1. Wczytanie modelu i obiektów
    model = load_model(model_path)
    with open(tok_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)

    # 2. Przetworzenie nowego tekstu dokładnie tak samo jak przy treningu
    seq = tokenizer.texts_to_sequences([text])
    X = pad_sequences(seq, maxlen=MAX_LEN)

    # 3. Predykcja
    pred = model.predict(X)[0]
    
    # 4. Interpretacja wyniku
    if len(pred) > 1:  # Dla wielu klas (np. Softmax)
        class_idx = np.argmax(pred)
        confidence = pred[class_idx]
    else:  # Dla dwóch klas (np. Sigmoid)
        confidence = pred[0] if pred[0] > 0.5 else 1 - pred[0]
        class_idx = 1 if pred[0] > 0.5 else 0

    label = le.inverse_transform([class_idx])[0]
    return f"{label} (Pewność: {confidence:.2f})"

def run_comparison(dataset_name, methods_list):
    """Uruchamia podane metody na zbiorze danych, generuje wykresy i zapisuje CSV."""
    texts, labels = load_sentiment_data(dataset_name)
    sample_size = min(100, len(texts))
    texts = texts[:sample_size]
    labels = labels[:sample_size]

    results = []

    for method in methods_list:
        y_pred = []
        print(f"Analiza metodą: {method}")

        for text in texts:
            try:
                if method == 'rule':
                    pred = sentiment_methods.analyze_rule_based(text)
                elif method == 'textblob':
                    pred = sentiment_methods.analyze_textblob(text)
                elif method == 'transformer':
                    pred, _ = sentiment_methods.analyze_transformer(text)
                elif method == 'stanza':
                    pred = sentiment_methods.analyze_stanza(text)
                elif method in ['simplernn', 'lstm', 'gru']:
                    raw_pred = predict_sentiment(text, method, dataset_name)
                    if "Błąd" in raw_pred:
                        pred = "błąd"
                    else:
                        pred = raw_pred.split(" ")[0].lower() # Wyciąga "pozytywny" z "pozytywny (Pewność: 0.8)"
                else:
                    pred = "nieznana"
            except Exception as e:
                print(f"Błąd dla {method}: {e}")
                pred = "błąd"

            y_pred.append(pred)

        if "błąd" in y_pred or "nieznana" in y_pred:
            continue

        acc = accuracy_score(labels, y_pred)
        prec = precision_score(labels, y_pred, average='macro', zero_division=0)
        rec = recall_score(labels, y_pred, average='macro', zero_division=0)
        f1 = f1_score(labels, y_pred, average='macro', zero_division=0)

        results.append({
            'dataset': dataset_name,
            'method': method,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'macro_f1': f1,
            'model_path': f"{MODELS_DIR}/{method}_{dataset_name}.h5" if method in ['simplernn', 'lstm', 'gru'] else 'brak'
        })

    # Zapis do pliku lab3results.csv
    csv_path = 'lab3results.csv'
    df = pd.DataFrame(results)
    df.to_csv(csv_path, mode='a', header=not os.path.isfile(csv_path), index=False)

    # Generowanie ładnego wykresu słupkowego
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results))
    width = 0.35

    accs = [r['accuracy'] for r in results]
    f1s = [r['macro_f1'] for r in results]
    method_names = [r['method'].upper() for r in results]

    plt.bar(x - width/2, accs, width, label='Accuracy', color='#3498db')
    plt.bar(x + width/2, f1s, width, label='Macro F1', color='#e74c3c')

    plt.ylabel('Wynik Metryki (0.0 - 1.0)')
    plt.title(f'Porównanie Modeli (Zbiór: {dataset_name.upper()}, Próbka: {sample_size})')
    plt.xticks(x, method_names)
    plt.legend()
    plt.ylim(0, 1.1)

    plot_path = os.path.join(PLOTS_DIR_LAB3, f"compare_methods_{dataset_name}.png")
    plt.savefig(plot_path)
    plt.close()

    # Formowanie odpowiedzi tekstowej do Telegrama
    raport = f"📊 **Podsumowanie porównania ({sample_size} próbek)**\n\n"
    for r in results:
        raport += f"🔹 **{r['method'].upper()}**\nAcc: `{r['accuracy']:.2f}` | F1: `{r['macro_f1']:.2f}`\n\n"
    raport += f"📁 Pełne dane (Precision, Recall) zapisano w `{csv_path}`."

    return raport, plot_path