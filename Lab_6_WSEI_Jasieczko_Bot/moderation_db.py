import csv
import os
from datetime import datetime

# Zgodnie z Lab06_2.md definiujemy nazwy plików
LOG_FILE = "moderation_log.csv"
USER_FILE = "user_moderation_history.csv"
FEEDBACK_FILE = "feedback_log.csv"

def init_databases():
    """Tworzy puste pliki bazy danych z nagłówkami, jeśli jeszcze nie istnieją."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "content_id", "user_id", "text", 
                "model_bielik_decision", "model_bielik_score", 
                "model_qwen_decision", "model_qwen_score", 
                "pii_detected", "sentiment", "action", 
                "moderator_override", "reason", "appeal_filed"
            ])
            
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "user_id", "username", "total_violations", 
                "last_violation_date", "categories", "risk_score", 
                "is_repeat_offender", "shadow_bans", "appeals_filed"
            ])
            
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "content_id", "original_bot_decision", "moderator_override", 
                "text_sample", "category", "confidence_before", 
                "confidence_after", "timestamp"
            ])

# Uruchamiamy inicjalizację przy starcie pliku
init_databases()

# =========================================================
# NARZĘDZIA MODERACYJNE I LOGIKA UŻYTKOWNIKÓW
# =========================================================

def log_moderation(content_id, user_id, text, bielik_dec, bielik_score, qwen_dec, qwen_score, pii, sentiment, action, reason):
    """Główna funkcja logująca każdą sprawdzoną wiadomość do pliku CSV."""
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(), content_id, user_id, text,
            bielik_dec, bielik_score, qwen_dec, qwen_score,
            pii, sentiment, action, "False", reason, "False"
        ])

def update_user_history(user_id: str, action: str, category: str, score: float):
    """Aktualizuje profil użytkownika w bazie (zapisuje naruszenia i status recydywisty)."""
    users = {}
    
    # Odczyt aktualnego stanu
    if os.path.exists(USER_FILE):
        with open(USER_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                users[row['user_id']] = row

    # Tworzenie nowego użytkownika, jeśli to jego pierwszy raz
    if user_id not in users:
        users[user_id] = {
            "user_id": user_id, "username": f"User_{user_id}", "total_violations": 0,
            "last_violation_date": "", "categories": "", "risk_score": 0.0,
            "is_repeat_offender": "False", "shadow_bans": 0, "appeals_filed": 0
        }

    user = users[user_id]
    
    # Jeśli bot odrzucił treść lub zbanował, rejestrujemy to jako naruszenie
    if action in ["REJECT", "BAN"]:
        user["total_violations"] = int(user["total_violations"]) + 1
        user["last_violation_date"] = datetime.now().isoformat()
        
        cats = user["categories"].split(";") if user["categories"] else []
        if category and category not in cats:
            cats.append(category)
        user["categories"] = ";".join(cats)
        
        user["risk_score"] = max(float(user["risk_score"]), score)
        
        if action == "BAN":
            user["shadow_bans"] = int(user["shadow_bans"]) + 1

        # Zasada recydywy: 3 lub więcej naruszeń
        if int(user["total_violations"]) >= 3:
            user["is_repeat_offender"] = "True"

    # Zapisz zaktualizowaną tabelę do pliku
    with open(USER_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "user_id", "username", "total_violations", "last_violation_date", 
            "categories", "risk_score", "is_repeat_offender", "shadow_bans", "appeals_filed"
        ])
        writer.writeheader()
        writer.writerows(users.values())
        
    return users[user_id]

# --- Narzędzia (Tools) dla Qwena ---

def approve_content(content_id: str, moderator_id: str = "bot") -> str:
    return f"✅ [ACTION] Wiadomość {content_id} zatwierdzona."

def reject_content(content_id: str, reason: str, moderator_id: str = "bot") -> str:
    return f"❌ [ACTION] Wiadomość {content_id} odrzucona. Powód: {reason}"

def flag_for_human_review(content_id: str, priority: str, reason: str) -> str:
    return f"⏳ [ACTION] Wiadomość {content_id} przekazana moderatorowi. Priorytet: {priority}. Powód: {reason}"

def shadow_ban_user(user_id: str, duration_hours: int, reason: str) -> str:
    return f"🚫 [ACTION] Użytkownik {user_id} zablokowany na {duration_hours}h. Powód: {reason}"

def add_feedback(content_id: str, original_decision: str, override: str, text_sample: str, category: str, confidence: float):
    """Narzędzie do uczenia modelu z decyzji człowieka."""
    with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            content_id, original_decision, override, text_sample, 
            category, confidence, confidence, datetime.now().isoformat()
        ])
    return f"✅ [FEEDBACK] Zapisano poprawkę dla {content_id}: {original_decision} -> {override}."