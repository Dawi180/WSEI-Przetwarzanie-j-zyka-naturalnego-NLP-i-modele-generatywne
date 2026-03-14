from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from data_handler import load_data

# Mapowanie klas na wartości liczbowe zdefiniowane w instrukcji
CLASS_MAPPING = {
    "pozytywny": 1,
    "neutralny": 0,
    "negatywny": -1
}

def train_and_predict(new_text):
    """
    Wczytuje bazę danych, trenuje prosty model ML i zwraca przewidzianą etykietę.
    """
    # 1. Wczytanie danych z pliku sentences.json
    data = load_data()
    
    # Zabezpieczenie: sprawdzamy, czy w ogóle mamy na czym się uczyć
    if len(data) < 2:
        return "Błąd: Za mało danych w 'sentences.json'. Użyj najpierw /task lub /full_pipeline, aby dodać więcej przykładów."
        
    texts = []
    labels = []
    
    # 2. Przygotowanie danych do uczenia modelu
    for item in data:
        texts.append(item.get("text", ""))
        labels.append(item.get("class", ""))
        
    # Sprawdzenie, czy mamy co najmniej dwie różne klasy (Logistic Regression tego wymaga)
    if len(set(labels)) < 2:
        return "Błąd: Model potrzebuje co najmniej dwóch różnych klas w bazie, aby móc się uczyć. Dodaj przykłady z innymi klasami."

    # 3. Zbudowanie prostego klasyfikatora tekstu (Pipeline)
    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    
    try:
        # Trenujemy model na wczytanych danych
        model.fit(texts, labels)
        
        # 4. Przewidzenie klasy dla nowej wiadomości
        prediction = model.predict([new_text])[0]
        
        # Wyciągamy wartość liczbową na podstawie mapowania
        numeric_value = CLASS_MAPPING.get(prediction.lower(), "Brak w standardowym mapowaniu")
        
        # 5. Zwracamy wynik
        return f"Przewidziana klasa: {prediction} (Wartość liczbowa: {numeric_value})"
        
    except Exception as e:
        return f"Wystąpił błąd podczas klasyfikacji: {str(e)}"