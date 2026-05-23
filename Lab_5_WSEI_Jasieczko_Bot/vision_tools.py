import requests
import base64
import os

OLLAMA_API_URL = "http://localhost:11434/api/generate"
# Używamy modelu pobranego w Kroku 1
VISION_MODEL = "minicpm-v" 

def analyze_image(image_path: str, prompt: str = "Opisz dokładnie, co widzisz na tym zdjęciu.") -> str:
    """
    Analizuje obraz za pomocą lokalnego modelu multimodalnego.
    Zwraca tekstowy opis zawartości obrazu.
    """
    print(f"👁️ [VISION] Analizuję obraz: {image_path}")
    
    if not os.path.exists(image_path):
        return f"❌ Błąd: Plik {image_path} nie istnieje."

    try:
        # LLM wymaga obrazu zakodowanego jako ciąg znaków w formacie Base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        payload = {
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": [encoded_string], # Lista obrazów (przekazujemy jeden)
            "stream": False,
            "options": {
                "temperature": 0.2 # Niska temperatura, żeby model trzymał się faktów na zdjęciu
            }
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "Nie udało się wygenerować opisu obrazu.")
        
    except requests.exceptions.ConnectionError:
        return "❌ Błąd połączenia: Upewnij się, że Ollama działa w tle."
    except Exception as e:
        return f"❌ Wystąpił nieoczekiwany błąd podczas analizy obrazu: {str(e)}"