import spacy
import stanza
from langdetect import detect

# ==========================================
# 1. ŁADOWANIE MODELI NLP
# ==========================================
print("Ładowanie modeli Spacy...")
try:
    nlp_spacy_en = spacy.load("en_core_web_sm")
    nlp_spacy_pl = spacy.load("pl_core_news_sm")
except OSError:
    print("Błąd: Nie znaleziono modeli Spacy. Upewnij się, że pobrałeś modele w terminalu!")

print("Ładowanie modeli Stanza (to może zająć chwilę za pierwszym razem)...")
# Stanza automatycznie pobierze brakujące modele dla j. polskiego i angielskiego
stanza.download('en', processors='tokenize,ner', verbose=False)
stanza.download('pl', processors='tokenize,ner', verbose=False)

nlp_stanza_en = stanza.Pipeline(lang='en', processors='tokenize,ner', verbose=False)
nlp_stanza_pl = stanza.Pipeline(lang='pl', processors='tokenize,ner', verbose=False)


# ==========================================
# 2. FUNKCJE ANALITYCZNE
# ==========================================
def detect_lang(text):
    """Wykrywa język tekstu (wsparcie dla wielu języków)."""
    try:
        # Zwracamy czysty wynik z biblioteki langdetect!
        return detect(text)
    except:
        return "en" # Bezpieczny fallback tylko w razie błędu

def perform_ner(text, method="spacy"):
    """
    Wykonuje Named Entity Recognition (NER).
    Zwraca sformatowany tekst gotowy do wysłania na Telegramie.
    """
    lang = detect_lang(text)
    entities = []

    if method == "spacy":
        nlp = nlp_spacy_pl if lang == 'pl' else nlp_spacy_en
        doc = nlp(text)
        for ent in doc.ents:
            # Format: - Nazwa (TYP) [start:end]
            entities.append(f"- {ent.text} ({ent.label_}) [{ent.start_char}:{ent.end_char}]")

    elif method == "stanza":
        nlp = nlp_stanza_pl if lang == 'pl' else nlp_stanza_en
        doc = nlp(text)
        for ent in doc.ents:
            entities.append(f"- {ent.text} ({ent.type}) [{ent.start_char}:{ent.end_char}]")

    else:
        return "❌ Nieznana metoda NER. Wybierz 'spacy' lub 'stanza'."

    # Budowanie ładnej odpowiedzi w stylu wymaganym w Lab 4
    if not entities:
        return f"Metoda: {method.capitalize()}\nJęzyk: {lang}\nTEXT: {text}\n\nENTITIES:\n- Nie znaleziono żadnych entitetów."

    result = (
        f"Metoda: {method.capitalize()}\n"
        f"Język: {lang}\n"
        f"TEXT: {text}\n\n"
        f"ENTITIES:\n" + "\n".join(entities)
    )
    return result