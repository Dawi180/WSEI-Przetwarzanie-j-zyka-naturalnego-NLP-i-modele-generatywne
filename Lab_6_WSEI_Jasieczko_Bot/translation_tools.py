from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect

_translators = {}

def get_translator(source_lang, target_lang):
    """Pobiera i ładuje do pamięci model tłumaczenia dla konkretnej pary językowej."""
    
    # Używamy potężnego modelu dla języków słowiańskich (w tym polskiego)
    if source_lang == "en" and target_lang == "pl":
        model_name = "Helsinki-NLP/opus-mt-en-sla"
    else:
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        
    if model_name not in _translators:
        print(f"⏳ Ładowanie modelu tłumaczenia: {model_name}... (może to potrwać przy pierwszym użyciu)")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            # DODANE: zapamiętujemy nazwę modelu, żeby użyć jej poniżej
            _translators[model_name] = {"tokenizer": tokenizer, "model": model, "name": model_name} 
        except Exception as e:
            print(f"❌ Nie udało się załadować modelu {model_name}. Błąd: {e}")
            return None
            
    return _translators[model_name]

def translate_text(text, target_lang, provided_source_lang=None):
    """
    Tłumaczy tekst na docelowy język przy użyciu natywnych modeli MarianMT.
    """
    # 1. Jeśli użytkownik wymusił język komendą (Manual Override), używamy go!
    if provided_source_lang:
        source_lang = provided_source_lang
    else:
        # 2. W przeciwnym razie bot zgaduje jak dawniej
        try:
            source_lang = detect(text)
        except:
            source_lang = "en"
            
    if source_lang == target_lang:
        return text, source_lang

    print(f"Tłumaczenie z {source_lang} na {target_lang}...")
    
    # 1. Próba tłumaczenia bezpośredniego
    translator_dict = get_translator(source_lang, target_lang)
    if translator_dict:
        tok = translator_dict["tokenizer"]
        mod = translator_dict["model"]
        model_name = translator_dict.get("name", "")
        
        # MAGIA MODELI WIELOJĘZYCZNYCH: dodajemy prefix 3-literowy
        input_text = text
        if "en-sla" in model_name:
            # Zamieniamy "pl" na oficjalny, trzyliterowy kod ISO: "pol"
            iso_lang = "pol" if target_lang == "pl" else target_lang
            input_text = f">>{iso_lang}<< {text}"
            
        # Tokenizacja, generowanie i dekodowanie
        inputs = tok(input_text, return_tensors="pt", padding=True, truncation=True)
        outputs = mod.generate(**inputs, max_length=512)
        decoded_batch = tok.batch_decode(outputs, skip_special_tokens=True)
        
        result = decoded_batch[0].strip() if decoded_batch else ""
        
        if not result:
            return "❌ Błąd dekodowania: Model zwrócił pusty tekst.", source_lang
            
        return result, source_lang
        
    # 2. Fallback: Tłumaczenie "z przesiadką" przez język angielski
    if source_lang != "en" and target_lang != "en":
        print(f"🔄 Brak bezpośredniego modelu {source_lang}-{target_lang}. Próba przez angielski...")
        trans_to_en = get_translator(source_lang, "en")
        trans_to_target = get_translator("en", target_lang)
        
        if trans_to_en and trans_to_target:
            # Etap 1: -> EN
            tok1, mod1 = trans_to_en["tokenizer"], trans_to_en["model"]
            inputs1 = tok1(text, return_tensors="pt", padding=True)
            out1 = mod1.generate(**inputs1, max_length=512)
            en_text = tok1.decode(out1[0], skip_special_tokens=True)
            
            # Etap 2: EN -> Cel
            tok2, mod2 = trans_to_target["tokenizer"], trans_to_target["model"]
            inputs2 = tok2(en_text, return_tensors="pt", padding=True)
            out2 = mod2.generate(**inputs2, max_length=512)
            final_text = tok2.decode(out2[0], skip_special_tokens=True)
            
            return final_text, source_lang
            
    return "❌ Błąd: Nie znaleziono obsługiwanego modelu dla tej pary językowej.", source_lang