import math
import json
import requests
from duckduckgo_search import DDGS
import re

# ==========================================
# 1. NARZĘDZIA LOKALNE (LOCAL TOOLS)
# ==========================================

def simple_calculator(expression: str) -> str:
    """
    Evaluate math expression.
    """
    print(f"🔧 [TOOL EXECUTED] simple_calculator: {expression}")
    
    try:
        # Zamiana √ na math.sqrt (na wszelki wypadek)
        expression = re.sub(r'√\s*\(?(\d+(?:\.\d+)?)\)?', r'math.sqrt(\1)', expression)
        expression = expression.replace('^', '**')
        
        # Udostępniamy konkretne funkcje z biblioteki math, w tym 'sqrt'
        allowed_names = {
            "math": math, 
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "pi": math.pi,
            "__builtins__": {}
        }
        
        result = eval(expression, allowed_names)
        return str(result)
    except Exception as e:
        return f"Błąd kalkulatora. Niepoprawne wyrażenie matematyczne: {str(e)}"
    
def local_knowledge(query: str) -> str:
    """
    Search in the local knowledge base for specific rules, internal information, or university details.
    """
    print(f"🔧 [TOOL EXECUTED] local_knowledge: {query}")
    
    knowledge_base = {
        "wsei": "Akademia WSEI to innowacyjna uczelnia wyższa.",
        "projekt": "Ten projekt to zaawansowany agent AI stworzony na zajęcia z Przetwarzania Języka Naturalnego.",
        "autor": "Autorem i głównym inżynierem tego bota jest Dawid Jasieczko.",
        "laboratorium": "Laboratorium 5 skupia się na wykorzystaniu Function Calling oraz analizy obrazów (Vision)."
    }
    
    # Zamieniamy przecinki na spacje i dzielimy na pojedyncze słowa
    query_words = query.lower().replace(",", " ").split()
    results = set() # Używamy zbioru (set), żeby unikać duplikatów
    
    for word in query_words:
        for key, value in knowledge_base.items():
            # Szukamy czy podane słowo pasuje do klucza
            if key in word or word in key:
                results.add(value)
                
    if results:
        return "Znalazłem w lokalnej bazie: " + " ".join(results)
    return "Brak informacji w lokalnej bazie na podany temat."

# ==========================================
# 2. NARZĘDZIA SIECIOWE (WEB TOOLS)
# ==========================================

def web_search(query: str) -> str:
    """
    Search the web for current events, news, or general knowledge.
    Use this tool when you need up-to-date information that is not in your training data.
    """
    print(f"🌐 [TOOL EXECUTED] web_search: {query}")
    try:
        with DDGS() as ddgs:
            # Pobieramy 3 najlepsze wyniki z DuckDuckGo
            results = list(ddgs.text(query, max_results=3))
            
        if not results:
            return "Brak wyników w internecie dla tego zapytania."
            
        # Złączenie fragmentów tekstu w jedno spójne podsumowanie
        summary = "\n".join([f"- {res['title']}: {res['body']}" for res in results])
        return f"Wyniki wyszukiwania w internecie:\n{summary}"
    except Exception as e:
        return f"Błąd podczas wyszukiwania w sieci: {str(e)}"

def get_weather(city: str) -> str:
    """
    Get the current weather forecast for a given city.
    Always use this tool when the user asks about the weather, temperature, or rain.
    """
    print(f"🌤️ [TOOL EXECUTED] get_weather: {city}")
    
    # Krok 1: Geokodowanie (Zamiana nazwy miasta na współrzędne geograficzne)
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=pl&format=json"
    try:
        geo_res = requests.get(geo_url).json()
        if "results" not in geo_res:
            return f"Nie mogłem znaleźć współrzędnych dla miasta: {city}."
            
        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]
        real_name = geo_res["results"][0]["name"]
        
        # Krok 2: Pobranie pogody dla współrzędnych (Darmowe API Open-Meteo)
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_res = requests.get(weather_url).json()
        
        weather = weather_res["current_weather"]
        temp = weather['temperature']
        wind = weather['windspeed']
        
        return f"Aktualna pogoda w {real_name}: {temp}°C, wiatr {wind} km/h."
    except Exception as e:
        return f"Błąd pobierania pogody dla {city}: {str(e)}"

# Słownik ułatwiający wywoływanie funkcji na podstawie ich nazwy
AVAILABLE_TOOLS = {
    "simple_calculator": simple_calculator,
    "local_knowledge": local_knowledge,
    "web_search": web_search,
    "get_weather": get_weather
}