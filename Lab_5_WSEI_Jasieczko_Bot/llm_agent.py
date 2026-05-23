import requests
import json
import agent_tools

OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5"

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "simple_calculator",
            "description": "Evaluate math expression. Use ONLY for math calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate, e.g., '2 + 2'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "local_knowledge",
            "description": "Search local knowledge base about WSEI, author, or project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword to search"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for news, facts, or current events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
]

def chat_with_agent(user_message: str):
    messages = [
        {"role": "system", "content": "Jesteś asystentem AI. Posiadasz dostęp do narzędzi. Zawsze podawaj wynik działania narzędzia (np. wynik matematyczny, pogodę). Nie wysyłaj kodu JSON użytkownikowi."},
        {"role": "user", "content": user_message}
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "tools": TOOLS_SCHEMA,
        "stream": False
    }

    try:
        print("🧠 [AGENT] Myślę nad zapytaniem...")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        response.encoding = 'utf-8'
        result = response.json()
        
        message = result.get("message", {})
        content = message.get("content", "").strip()
        tool_calls = message.get("tool_calls", [])
        
        # Ochrona przed wyciekiem JSON (częste dla kalkulatora)
        if not tool_calls and content.startswith("{") and '"parameters"' in content:
            print("🔧 [FALLBACK] Przechwycono wyciek JSON w tekście! Wymuszam narzędzie...")
            try:
                fake_call = json.loads(content)
                func_name = "simple_calculator" if "math" in fake_call.get("name", "").lower() else fake_call.get("name")
                args = fake_call.get("parameters", {})
                tool_calls = [{"function": {"name": func_name, "arguments": args}}]
                message["content"] = "" 
            except Exception:
                pass

        if tool_calls:
            messages.append(message) 
            
            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]
                
                print(f"🤖 [MODEL DECISION] Używam narzędzia: {func_name} z argumentami {args}")
                
                if func_name in agent_tools.AVAILABLE_TOOLS:
                    tool_result = agent_tools.AVAILABLE_TOOLS[func_name](**args)
                else:
                    tool_result = f"Error: Tool {func_name} not found."
                    
                messages.append({
                    "role": "tool",
                    "content": str(tool_result),
                    "name": func_name
                })
                
            print("🧠 [AGENT QWEN] Analizuję zebrane wyniki i piszę piękną odpowiedź...")
            
            # Instrukcja zmuszająca do użycia WSZYSTKICH zebranych danych
            messages.append({
                "role": "system",
                "content": "Teraz odpowiedz zwięźle użytkownikowi, łącząc powyższe informacje z narzędzi w spójne zdania. Jeśli użyłeś kilku narzędzi (np. pogoda i kalkulator), musisz podać wyniki z nich wszystkich."
            })

            second_payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.4 # Qwen lubi tę temperaturę do gładkiego pisania
                }
            }
            
            final_response = requests.post(OLLAMA_API_URL, json=second_payload, timeout=120)
            final_response.encoding = 'utf-8' # Zabezpieczenie polskich znaków w odpowiedzi z narzędzi
            final_response.raise_for_status()
            
            final_content = final_response.json().get("message", {}).get("content", "").strip()
            
            if not final_content:
                 return "Oto suche fakty, bo model nie umiał złożyć z tego zdania:\n" + "\n".join([m['content'] for m in messages if m.get('role') == 'tool'])
            
            return final_content
            
        else:
            print("🤖 [MODEL DECISION] Odpowiadam bezpośrednio (bez narzędzi).")
            return content if content else "❌ [Błąd] Model nie zwrócił żadnego tekstu."

    except Exception as e:
        return f"❌ Błąd komunikacji z agentem Ollama: {str(e)}"