import json
import uuid
import asyncio
from datetime import datetime
from pydantic import BaseModel
from ollama import AsyncClient

# Importujemy nasze własne moduły moderacyjne
import moderation_tools
import moderation_db
# Zakładam, że narzędzia NLP (sentyment, NER) masz z poprzednich labów, np. w nlp_tools.py
# W razie braku, użyjemy prostych placeholderów
try:
    import nlp_tools
except ImportError:
    nlp_tools = None

# Inicjalizacja Ollama i definicja modelu
ollama_client = AsyncClient()
QWEN_MODEL = "qwen2.5"

# Szablon systemu dla naszego Moderatora Qwen
SYSTEM_PROMPT_MODERATOR = """
Jesteś zaawansowanym asystentem ds. moderacji treści.
Twoim zadaniem jest ocena potencjalnie szkodliwych tekstów i wywoływanie narzędzi moderacyjnych.

Dostępne akcje to:
- "approve_content": Zawsze używaj, jeśli tekst jest bezpieczny i czysty. Zgłoszenia "negative_sentiment" nie są powodem do odrzucenia (ludzie mogą narzekać, dopóki nikogo nie obrażają).
- "reject_content": Używaj przy wulgaryzmach, hejcie, spamie lub gdy wykryto dane wrażliwe (PII). Podaj krótki powód ("reason").
- "flag_for_human_review": Używaj, gdy tekst jest bardzo niejednoznaczny (np. ostra opinia polityczna) i wymagasz decyzji człowieka. Podaj priorytet (low, medium, high).
- "shadow_ban_user": Używaj TYLKO przy skrajnym, brutalnym hejcie lub powtarzającym się, agresywnym spamie.

Przeanalizuj tekst użytkownika oraz wyniki innych modeli i wybierz jedno najlepsze narzędzie do wykonania.
"""

def extract_qwen_tools():
    """Definicja schematu narzędzi dla Qwena 2.5 (Function Calling)"""
    return [
        {
            "type": "function",
            "function": {
                "name": "approve_content",
                "description": "Zatwierdzanie czystej, bezpiecznej zawartości (nawet negatywnych opinii klientów).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_id": {"type": "string"},
                        "moderator_id": {"type": "string"}
                    },
                    "required": ["content_id", "moderator_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "reject_content",
                "description": "Odrzucanie wulgaryzmów, hejtu, spamu lub tekstów zawierających PII.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_id": {"type": "string"},
                        "reason": {"type": "string", "description": "Powód odrzucenia"},
                        "moderator_id": {"type": "string"}
                    },
                    "required": ["content_id", "reason", "moderator_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "flag_for_human_review",
                "description": "Przekazanie skomplikowanych spraw człowiekowi.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_id": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                        "reason": {"type": "string", "description": "Dlaczego sprawa jest wątpliwa?"}
                    },
                    "required": ["content_id", "priority", "reason"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "shadow_ban_user",
                "description": "Opcja ostateczna (ban). Używać tylko przy krytycznym hejcie.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "duration_hours": {"type": "integer"},
                        "reason": {"type": "string"}
                    },
                    "required": ["user_id", "duration_hours", "reason"]
                }
            }
        }
    ]

async def process_message(text: str, user_id: str) -> str:
    """
    Główna rura przetwarzania (Pipeline) dla Laboratorium 6.
    Łączy modele bezpieczeństwa, decyduje o akcji i loguje ją.
    """
    content_id = f"MSG-{str(uuid.uuid4())[:8].upper()}"
    bot_response = f"**Identyfikator wiadomości:** `{content_id}`\n\n"
    
    # KROK 1: Klasyfikacja modelami zewnętrznymi
    pii_results = moderation_tools.detect_private_info(text)
    bielik_results = moderation_tools.classify_bielik_guard(text)
    
    # --- NOWE: Analiza NLP z Lab 3 i 4 ---
    sentiment_data = nlp_tools.analyze_sentiment_for_moderation(text)
    ner_data = nlp_tools.extract_moderation_entities(text)
    sentiment_result = sentiment_data['sentiment']
    
    bot_response += f"🔍 **Wyniki Modeli Ochronnych:**\n"
    bot_response += f"- Bielik Guard: `{bielik_results['label']}` (pewność: {bielik_results['score']:.2f})\n"
    bot_response += f"- Detekcja PII: `{'WYKRYTO!' if pii_results['has_pii'] else 'Brak'}`\n"
    bot_response += f"- Sentyment: `{sentiment_result.upper()}` (Emocja: {sentiment_data['emotion']})\n"
    
    # Wyświetlamy znalezione encje, jeśli jakieś są
    found_entities = [item for sublist in ner_data.values() for item in sublist]
    if found_entities:
         bot_response += f"- Wykryte obiekty (NER): `{', '.join(found_entities)}`\n"
    bot_response += "\n"

    # KROK 2: Przygotowanie "Raportu" dla Qwena
    qwen_prompt = f"""
Przeanalizuj poniższą wiadomość od użytkownika: '{user_id}'. ID Treści: '{content_id}'.

TREŚĆ WIADOMOŚCI:
"{text}"

WYNIKI MODELI POMOCNICZYCH:
- Wrażliwe dane (PII): {'TAK' if pii_results['has_pii'] else 'NIE'}
- Klasyfikator zagrożeń: Etykieta '{bielik_results['label']}'
- Sentyment tekstu: {sentiment_result}
- Wykryte obiekty/linki/osoby: {found_entities}

ZASADY:
1. Jeśli PII to 'TAK' -> MUSISZ wezwać reject_content.
2. Jeśli Bielik wykrył hejt/spam -> rozważ reject_content.
3. Jeśli tekst ma sentyment 'negative', ale to tylko narzekanie na produkt/usługę (bez hejtu i wyzwisk) -> wezwij approve_content.
4. Użyj flag_for_human_review, jeśli masz duże wątpliwości lub wspomniano o polityce.
"""

    bot_response += "🤖 **Analiza przez LLM (Qwen 2.5)...**\n"
    
    # KROK 3: Uruchomienie modelu Qwen2.5 w trybie Function Calling
    try:
        response = await ollama_client.chat(
            model=QWEN_MODEL,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT_MODERATOR},
                {'role': 'user', 'content': qwen_prompt}
            ],
            tools=extract_qwen_tools(),
            options={'temperature': 0.1} # Niska temp., chcemy powtarzalności
        )
        
        # Odbieranie decyzji (wybranego narzędzia)
        message = response['message']
        qwen_decision = "approved"
        qwen_score = 0.95 # Przy LLM ciężko o hard-score, dajemy wartość domyślną
        final_action = "APPROVE"
        final_reason = ""
        action_log = "Zatwierdzono (domyślnie)."
        
        if message.get('tool_calls'):
            tool_call = message['tool_calls'][0]
            function_name = tool_call['function']['name']
            args = tool_call['function']['arguments']
            
            # Mapowanie funkcji na ostateczną decyzję do CSV
            if function_name == "reject_content":
                qwen_decision = "rejected"
                final_action = "REJECT"
                final_reason = args.get('reason', 'N/A')
                action_log = moderation_db.reject_content(content_id, final_reason)
            elif function_name == "shadow_ban_user":
                qwen_decision = "banned"
                final_action = "BAN"
                final_reason = args.get('reason', 'N/A')
                action_log = moderation_db.shadow_ban_user(user_id, args.get('duration_hours', 24), final_reason)
            elif function_name == "flag_for_human_review":
                qwen_decision = "flagged"
                final_action = "REVIEW"
                final_reason = args.get('reason', 'N/A')
                action_log = moderation_db.flag_for_human_review(content_id, args.get('priority', 'medium'), final_reason)
            elif function_name == "approve_content":
                qwen_decision = "approved"
                final_action = "APPROVE"
                action_log = moderation_db.approve_content(content_id)
                
            bot_response += f"✓ **Akcja:** {action_log}\n"
        else:
            bot_response += "⚠️ Qwen nie wybrał żadnego narzędzia (fallback: Approve).\n"

        # KROK 4: Logowanie decyzji do bazy (Zapis do CSV)
        moderation_db.log_moderation(
            content_id=content_id,
            user_id=user_id,
            text=text,
            bielik_dec=bielik_results['label'],
            bielik_score=bielik_results['score'],
            qwen_dec=qwen_decision,
            qwen_score=qwen_score,
            pii=pii_results['has_pii'],
            sentiment=sentiment_result,
            action=final_action,
            reason=final_reason
        )

        # KROK 5: Aktualizacja profilu użytkownika (Rejestr naruszeń)
        # Bierzemy kategorię z Bielika, chyba że uważa, że jest czysto (wtedy bierzemy decyzję Qwena)
        violation_category = bielik_results['label'] if bielik_results['label'] != 'clean' else qwen_decision
        
        user_profile = moderation_db.update_user_history(
            user_id=user_id,
            action=final_action,
            category=violation_category,
            score=bielik_results['score']
        )
        
        # Ostrzeżenie, jeśli wykryto recydywę
        if user_profile.get("is_repeat_offender") == "True" and final_action in ["REJECT", "BAN"]:
            bot_response += f"\n⚠️ **ALERT ADMINA:** Użytkownik `{user_id}` jest oznaczony jako recydywista (Repeat Offender)!"

        return bot_response

    except Exception as e:
         return f"❌ Błąd w silniku moderacyjnym Qwena: {e}"

# Test dla konsoli
if __name__ == "__main__":
    async def run_test():
        print(await process_message("Ten produkt to totalne śmieci, nienawidzę was. Mój pesel to 123456789.", "User777"))
    asyncio.run(run_test())