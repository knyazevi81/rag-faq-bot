import re
from transformers import pipeline

def preprocess_text(text: str) -> str:
    """Предварительная обработка текста"""
    # Сшивка переносов слов
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Удаление лишних пробелов
    text = re.sub(r"\s+", " ", text)
    # Удаление специфичных артефактов PDF
    text = re.sub(r"Adobe PDF Library|Layout by [A-Z\.]+", "", text)
    return text.strip()

def extract_entities(text: str) -> list:
    """Извлечение именованных сущностей из текста"""
    try:
        ner = pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner")
        entities = ner(text)
        # Собираем только сущности с высокой уверенностью
        return list(set([e['word'] for e in entities if e['score'] > 0.8]))
    except Exception as e:
        print(f"Entity extraction error: {e}")
        return []