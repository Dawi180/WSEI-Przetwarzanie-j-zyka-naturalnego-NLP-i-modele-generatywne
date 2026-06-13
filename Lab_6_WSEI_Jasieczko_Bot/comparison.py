import os
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import dl_models
import sentiment_methods
from config import PLOTS_DIR_LAB3

def run_comparison(dataset_name, methods_list):
    """Przeprowadza porównanie wielu modeli na jednym datasetcie."""
    # 1. Pobieranie danych
    texts, labels = dl_models.load_sentiment_data(dataset_name)

    # Ograniczamy zbiór do 100 losowych próbek, by nie czekać na Transformery godzinami
    sample_size = min(100, len(texts))
    combined = list(zip(texts, labels))
    random.seed(42)
    random.shuffle(combined)
    texts, labels = zip(*combined[:sample_size])
    texts, labels = list(texts), list(labels)

    results = []

    # 2. Ewaluacja każdej wybranej metody
    for method in methods_list:
        print(f"Rozpoczynam ewaluację metody: {method}...")
        y_pred = []
        
        for text in texts:
            if method == "rule":
                pred = sentiment_methods.analyze_rule_based(text)
            elif method == "textblob":
                pred = sentiment_methods.analyze_textblob(text)
            elif method == "transformer":
                pred, _ = sentiment_methods.analyze_transformer(text)
            elif method == "stanza":
                pred = sentiment_methods.analyze_stanza(text)
            elif method in ["simplernn", "lstm", "gru"]:
                # Funkcja z dl_models zwraca np. "pozytywny (Pewność: 0.85)"
                # My musimy wyciągnąć z tego tylko pierwsze słowo: "pozytywny"
                raw_pred = dl_models.predict_sentiment(text, method, dataset_name)
                pred = raw_pred.split(" ")[0].lower()
            else:
                pred = "neutralny"
                
            y_pred.append(pred)

        # 3. Obliczanie zaawansowanych metryk ML
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
            'model_path': f"models/{method}_{dataset_name}.h5" if method in ['simplernn', 'lstm', 'gru'] else "N/A"
        })

    # 4. Zapis do pliku CSV
    df = pd.DataFrame(results)
    csv_path = 'lab3results.csv'
    file_exists = os.path.isfile(csv_path)
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

    # 5. Generowanie Wykresu Porównawczego
    plt.figure(figsize=(10, 6))
    methods_names = df['method'].tolist()
    accuracies = df['accuracy'].tolist()
    f1_scores = df['macro_f1'].tolist()

    x = range(len(methods_names))
    width = 0.35

    plt.bar([i - width/2 for i in x], accuracies, width, label='Accuracy', color='skyblue')
    plt.bar([i + width/2 for i in x], f1_scores, width, label='Macro F1', color='salmon')

    plt.xlabel('Metody Analizy Sentymentu')
    plt.ylabel('Wynik Metryki (0.0 - 1.0)')
    plt.title(f'Porównanie Modeli (Zbiór: {dataset_name.upper()}, Próbka: {sample_size})')
    plt.xticks(x, methods_names)
    plt.legend()
    plt.ylim(0, 1.1)

    plot_path = os.path.join(PLOTS_DIR_LAB3, f"compare_methods_{dataset_name}.png")
    plt.savefig(plot_path)
    plt.close()

    # 6. Formatowanie ładnego raportu na Telegrama
    report = f"📊 **Podsumowanie porównania ({sample_size} próbek)**\n\n"
    for r in results:
        report += f"🔹 `{r['method'].upper()}`\n  Acc: **{r['accuracy']:.2f}** | F1: **{r['macro_f1']:.2f}**\n\n"
    
    report += "📁 Pełne dane (Precision, Recall) zapisano w `lab3results.csv`."

    return report, plot_path