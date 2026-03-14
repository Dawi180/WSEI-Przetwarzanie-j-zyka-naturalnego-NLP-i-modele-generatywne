import json
import os

FILE_NAME = 'sentences.json'

def init_db():
    """Tworzy pusty plik JSON z pustą listą, jeśli jeszcze nie istnieje."""
    if not os.path.exists(FILE_NAME):
        with open(FILE_NAME, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)

def load_data():
    """Wczytuje dane z pliku JSON."""
    if not os.path.exists(FILE_NAME):
        init_db()
    with open(FILE_NAME, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            # Zwróć pustą listę, jeśli plik jest pusty lub uszkodzony
            return []

def save_record(text, text_class):
    """
    Dopisuje nowy rekord do pliku JSON.
    Zgodnie z założeniami dodaje tylko pola 'text' i 'class'.
    """
    data = load_data()
    
    # Tworzymy nowy rekord dokładnie w takim formacie, jak w instrukcji
    new_record = {
        "text": text,
        "class": text_class
    }
    
    data.append(new_record)
    
    # Zapisujemy z powrotem do pliku
    with open(FILE_NAME, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_classes_stats():
    """Zwraca statystyki występowania klas (do komendy /stats)."""
    data = load_data()
    stats = {}
    for item in data:
        label = item.get("class")
        if label:
            stats[label] = stats.get(label, 0) + 1
    return stats