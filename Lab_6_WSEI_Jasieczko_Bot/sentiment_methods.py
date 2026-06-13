from textblob import TextBlob
from transformers import pipeline
import stanza

# Pobieranie paczki językowej Stanza (wykona się tylko raz)
print("Ładowanie modelu Stanza (to może zająć chwilę za pierwszym razem)...")
stanza.download('en', processors='tokenize,sentiment')
stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment', verbose=False)

# Ładowanie modelu Transformer
print("Ładowanie modelu Transformer...")
hf_analyzer = pipeline("sentiment-analysis")

def analyze_rule_based(text):
    """Prymitywna analiza oparta o słowa kluczowe (Rule-based)."""
    pozytywne = ["dobry", "świetny", "super", "uwielbiam", "polecam", "fajny", "najlepszy"]
    negatywne = ["zły", "słaby", "okropny", "fatalny", "nie polecam", "beznadziejny", "szkoda"]

    text_lower = text.lower()
    score = 0
    
    for word in pozytywne:
        if word in text_lower: score += 1
    for word in negatywne:
        if word in text_lower: score -= 1

    if score > 0: return "pozytywny"
    elif score < 0: return "negatywny"
    else: return "neutralny"

def analyze_textblob(text):
    """Analiza przy użyciu biblioteki TextBlob."""
    # TextBlob natywnie działa najlepiej dla j. angielskiego, ale jako model bazowy wystarczy
    analysis = TextBlob(text)
    
    if analysis.sentiment.polarity > 0.1:
        return "pozytywny"
    elif analysis.sentiment.polarity < -0.1:
        return "negatywny"
    else:
        return "neutralny"
    
def analyze_transformer(text):
    """Analiza za pomocą Transformera z biblioteki HuggingFace."""
    result = hf_analyzer(text)[0]
    # Transformery zazwyczaj zwracają etykiety 'POSITIVE' lub 'NEGATIVE'
    if result['label'] == 'POSITIVE':
        return "pozytywny", result['score']
    else:
        return "negatywny", result['score']

def analyze_stanza(text):
    """Analiza za pomocą biblioteki Stanza od Stanford University."""
    doc = stanza_nlp(text)
    # Stanza ocenia każde zdanie z osobna: 0 (negatywne), 1 (neutralne), 2 (pozytywne)
    total_sentiment = sum([sentence.sentiment for sentence in doc.sentences])
    avg_sentiment = total_sentiment / len(doc.sentences) if len(doc.sentences) > 0 else 1
    
    if avg_sentiment > 1.5:
        return "pozytywny"
    elif avg_sentiment < 0.5:
        return "negatywny"
    else:
        return "neutralny"