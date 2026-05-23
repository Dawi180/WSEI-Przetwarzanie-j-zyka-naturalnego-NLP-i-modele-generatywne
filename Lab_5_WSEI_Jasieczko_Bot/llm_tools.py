import requests
import time

# Domyślny adres lokalnego API Ollamy
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "phi3"  # Ustawiamy model, który pobrałeś w terminalu

def generate_summary(text, summary_type="abstractive", length="medium"):
    """
    Generuje podsumowanie tekstu przy użyciu lokalnego modelu LLM przez Ollama API.
    """
    # 1. Konfiguracja długości (Prompt Engineering)
    if length == "short":
        len_prompt = "w 2-3 krótkich zdaniach."
    elif length == "long":
        len_prompt = "w 3-4 szczegółowych akapitach."
    else:
        len_prompt = "w około 1 zwięzłym akapicie."

    # 2. Konfiguracja typu podsumowania
    if summary_type == "extractive":
        sys_prompt = f"Wybierz i zacytuj najważniejsze, oryginalne zdania z poniższego tekstu {len_prompt} Nie zmieniaj ich treści."
    elif summary_type == "bullets":
        sys_prompt = f"Wypisz w punktach najważniejsze informacje z poniższego tekstu {len_prompt}"
    else:  # abstractive
        sys_prompt = f"Napisz własnymi słowami zwięzłe podsumowanie poniższego tekstu {len_prompt}"

    # Ostateczny prompt wysyłany do modelu
    full_prompt = f"Jesteś profesjonalnym asystentem. {sys_prompt}\n\nTekst:\n{text}"

    # Parametry dla Ollama API
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": full_prompt,
        "stream": False,  # Chcemy dostać całą odpowiedź na raz, a nie po słowie
        "options": {
            "temperature": 0.3  # Niska temperatura = bardziej rzeczowy, mniej kreatywny tekst
        }
    }

    try:
        # 3. Wysłanie zapytania z obsługą Timeoutu (Wymóg Lab 4)
        start_time = time.time()
        
        # Czekamy max 120 sekund na wygenerowanie odpowiedzi
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        gen_time = time.time() - start_time
        data = response.json()
        
        summary = data.get("response", "Brak odpowiedzi od modelu.")
        
        # Zgrubne liczenie tokenów (długość tekstu w słowach)
        text_length = len(text.split())
        
        # 4. Budowanie ładnego formatu zgodnego z wymaganiami z instrukcji
        result = (
            f"Model: {DEFAULT_MODEL.capitalize()}\n"
            f"Text length: ~{text_length} tokens\n"
            f"Summary type: {summary_type.capitalize()}\n"
            f"Summary length: {length.capitalize()}\n\n"
            f"SUMMARY:\n{summary}\n\n"
            f"Generation time: {gen_time:.2f}s"
        )
        return result
        
    except requests.exceptions.ConnectionError:
        return "❌ Błąd połączenia: Upewnij się, że aplikacja Ollama jest uruchomiona w tle na Twoim komputerze!"
    except requests.exceptions.ReadTimeout:
        return "⏳ Błąd: Model przetwarzał tekst zbyt długo (Timeout). Spróbuj na krótszym tekście."
    except Exception as e:
        return f"❌ Wystąpił nieoczekiwany błąd Ollamy: {str(e)}"