---
name: runeterra-lore-rag
description: >-
  Expert assistant and operations skill for the Runeterra Lore RAG project.
  Guides web lore extraction (Data Dragon, League Wiki, Universe stories),
  local book ingestion (Realms of Runeterra, Ruination), hybrid retrieval
  (BM25 + BGE-M3 + Cross-Encoder Reranker), and question answering with Gemini.
---

# Runeterra Lore RAG - Operation & Development Skill

Use this skill to inspect, extract, run, evaluate, or expand the Runeterra Lore RAG system.

---

## 🛠️ Key Capabilities & Common Commands

### 1. Web Lore Extraction (`src/lore_extractor.py`)
Run the web extractor using `uv run python src/lore_extractor.py` with appropriate flags:
- Extract specific champion (pt_BR or en_US):
  ```bash
  uv run python src/lore_extractor.py --champion Viego --lang pt_BR
  ```
- Extract all 170+ champion biographies:
  ```bash
  uv run python src/lore_extractor.py --champions --lang pt_BR
  ```
- Extract Region Lore overviews (Demacia, Noxus, Ionia, etc.):
  ```bash
  uv run python src/lore_extractor.py --regions
  ```
- Extract official Universe short stories from League Wiki:
  ```bash
  uv run python src/lore_extractor.py --stories --limit-stories 5
  ```

### 2. Testing Retrieval & Answering (`src/rag_engine.py`)
To test retrieval programmatically:
```python
from src.rag_engine import RuneterraRAGEngine
from src.book_extractor import BookExtractor

engine = RuneterraRAGEngine()
extractor = BookExtractor()
pages = extractor.extract_all_books()
engine.ingest(pages=pages, include_web_lore=True)

answer, sources = engine.answer_question("Quem foi o primeiro governante de Camavor?")
print("Answer:", answer)
print("Sources:", sources)
```

### 3. Running the Gradio Web Interface
```bash
uv run python src/app.py
```

---

## 📂 Project Organization & Paths
- **Config & Environment**: Loaded via `src/config.py` from `.env`.
- **Books**: Stored in `data/books/` (PDF files).
- **Web Lore**: Stored in `data/web_lore/` (subfolders `champions/`, `regions/`, `stories/`).
- **Cache**: Embeddings `.npy` and OCR JSON cached in `data/cache/`.
- **Notebook**: `runeterra_rag_v2.ipynb` contains an end-to-end interactive version with auto-detection for Google Colab vs Local execution.
