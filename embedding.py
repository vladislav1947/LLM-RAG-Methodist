import os
import re
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# === Конфигурация ===
INPUT_FILE = Path("path/to/split_document.txt")  # Путь к файлу с уже разбитыми фрагментами
SEPARATOR = r"=+\s*"                             # Регулярное выражение-разделитель
OUTPUT_DIR = Path("embeddings")                  # Куда сохранить результат
LABEL = "your_dataset_name"                      # Название набора для имени файла
MODEL_NAME = "your-model-name"                   # Название модели-эмбеддера с HuggingFace

# === Загрузка модели эмбеддингов ===
model = SentenceTransformer(MODEL_NAME)

# === Убедимся, что выходная папка существует ===
OUTPUT_DIR.mkdir(exist_ok=True)

# === Загрузка и разбор входного файла ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Разделяем по указанному разделителю
fragments = [frag.strip() for frag in re.split(SEPARATOR, raw_text) if frag.strip()]

if not fragments:
    raise ValueError("Нет подходящих фрагментов для векторизации. Проверьте формат или минимальную длину.")

# === Генерация эмбеддингов ===
embeddings = model.encode(fragments, show_progress_bar=True, convert_to_numpy=True)

# === Создание и сохранение FAISS индекса ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss_index_path = OUTPUT_DIR / f"faiss_{LABEL}.index"
metadata_path = OUTPUT_DIR / f"metadata_{LABEL}.pkl"

faiss.write_index(index, str(faiss_index_path))

# Метаданные: только текст и путь к источнику
with open(metadata_path, "wb") as f:
    pickle.dump({
        "documents": fragments,
        "sources": [str(INPUT_FILE)] * len(fragments)
    }, f)

print(f"Сохранено: {faiss_index_path.name} ({len(fragments)} фрагментов)")
