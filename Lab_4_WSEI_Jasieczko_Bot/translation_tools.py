from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect

_translators = {}

def get_translator(source_lang, target_lang):
    """Pobiera i ładuje do pamięci model tłumaczenia dla konkretnej pary językowej."""
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    
    if model_name not in _translators:
        print(f"⏳ Ładowanie modelu tłumaczenia: {model_name}... (może to potrwać przy pierwszym użyciu)")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            _translators[model_name] = {"tokenizer": tokenizer, "model": model}
        except Exception as e:
            print(f"❌ Nie udało się załadować modelu {model_name}. Błąd: {e}")
            return None
            
    return _translators[model_name]

def translate_text(text, target_lang):
    """
    Tłumaczy tekst na docelowy język przy użyciu natywnych modeli MarianMT.
    """
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
        
        # Tokenizacja, generowanie i dekodowanie
        inputs = tok(text, return_tensors="pt", padding=True)
        outputs = mod.generate(**inputs, max_new_tokens=512)
        result = tok.decode(outputs[0], skip_special_tokens=True)
        
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
            out1 = mod1.generate(**inputs1, max_new_tokens=512)
            en_text = tok1.decode(out1[0], skip_special_tokens=True)
            
            # Etap 2: EN -> Cel
            tok2, mod2 = trans_to_target["tokenizer"], trans_to_target["model"]
            inputs2 = tok2(en_text, return_tensors="pt", padding=True)
            out2 = mod2.generate(**inputs2, max_new_tokens=512)
            final_text = tok2.decode(out2[0], skip_special_tokens=True)
            
            return final_text, source_lang
            
    return "❌ Błąd: Nie znaleziono obsługiwanego modelu dla tej pary językowej.", source_lang