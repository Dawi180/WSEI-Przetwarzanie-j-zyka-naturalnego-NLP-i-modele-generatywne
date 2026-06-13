import os
from dotenv import load_dotenv
from transformers import pipeline

# Załadowanie zmiennych z pliku .env (w tym HF_TOKEN)
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

print("🛡️ [MODERACJA] Ładowanie lokalnych modeli bezpieczeństwa...")

# Słownik do przechowywania załadowanych modeli w pamięci
_guard_models = {}

def load_security_models():
    """Ładuje modele do pamięci przy pierwszym uruchomieniu."""
    if "pii" not in _guard_models:
        try:
            _guard_models["pii"] = pipeline("token-classification", model="openai/privacy-filter")
            print("✅ [MODERACJA] Załadowano model Privacy Filter")
        except Exception as e:
            print(f"❌ [MODERACJA] Błąd ładowania Privacy Filter: {e}")
            _guard_models["pii"] = None

    if "bielik" not in _guard_models:
        try:
            # Model 2: Detekcja toksyczności (Z DODANYM TOKENEM)
            if not hf_token:
                print("❌ [MODERACJA] Brak HF_TOKEN w pliku .env! Bielik nie może zostać pobrany.")
                _guard_models["bielik"] = None
            else:
                _guard_models["bielik"] = pipeline(
                    "text-classification", 
                    model="speakleash/Bielik-Guard-0.1B-v1.0",
                    token=hf_token  # To odblokuje repozytorium!
                )
                print("✅ [MODERACJA] Załadowano model Bielik Guard")
        except Exception as e:
            print(f"❌ [MODERACJA] Błąd ładowania Bielik Guard: {e}")
            _guard_models["bielik"] = None

# Wywołujemy ładowanie na starcie
load_security_models()

def detect_private_info(text: str) -> dict:
    """
    Detekcja poufnych informacji (SSN, karty kredytowe, adresy email).
    """
    pii_pipeline = _guard_models.get("pii")
    if not pii_pipeline:
        return {'has_pii': False, 'entities': []}
    
    results = pii_pipeline(text)
    
    # Parsowanie wyników z modelu Token Classification
    entities = [{'type': res['entity'], 'word': res['word'], 'score': float(res['score'])} for res in results]
    
    return {
        'has_pii': len(entities) > 0,
        'entities': entities
    }

def classify_bielik_guard(text: str) -> dict:
    """
    Klasyfikacja tekstu pod kątem zagrożeń z confidence score.
    """
    bielik_pipeline = _guard_models.get("bielik")
    if not bielik_pipeline:
        return {'label': 'clean', 'score': 1.0, 'severity': 'low'}
        
    # Uruchomienie modelu text-classification
    results = bielik_pipeline(text)
    
    if not results:
        return {'label': 'clean', 'score': 1.0, 'severity': 'low'}
        
    top_result = results[0]
    label = top_result['label'].lower()
    score = float(top_result['score'])
    
    # Ustalanie poziomu zagrożenia (severity) na podstawie pewności modelu (score)
    if label == 'clean' or label == 'safe':
        severity = 'low'
    elif score >= 0.90:
        severity = 'critical'
    elif score >= 0.70:
        severity = 'high'
    elif score >= 0.40:
        severity = 'medium'
    else:
        severity = 'low'
        
    return {
        'label': label,
        'score': score,
        'severity': severity
    }

# --- TESTOWANIE ---
if __name__ == "__main__":
    test_tekst = "Nienawidzę tego produktu! Mój numer telefonu to 123-456-789."
    print("\n[TEST] Analiza tekstu:", test_tekst)
    print("PII:", detect_private_info(test_tekst))
    print("Bielik Guard:", classify_bielik_guard(test_tekst))