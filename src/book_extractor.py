"""
Book and Document Extractor for Runeterra Lore RAG.
Extracts native text and runs OCR on illustrated/scanned pages from PDF lore books.
Caches processed pages to JSON for instant reloads.
"""

import io
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
from PIL import Image
from pypdf import PdfReader
from tqdm import tqdm

from src.config import BOOKS_DIR, CACHE_DIR, get_book_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BookExtractor")

MIN_CHARS = 80      # Pages with less text than this go through OCR
OCR_DPI = 200       # Higher = better accuracy, slower
BATCH = 16          # Pages rendered per batch


import shutil

def is_tesseract_available() -> bool:
    """Checks whether the Tesseract-OCR binary is available on the system PATH."""
    if shutil.which("tesseract"):
        return True
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except (ImportError, OSError):
        return False


def clean_book_title(path: str) -> str:
    """Formats clean book title from filename."""
    name = os.path.splitext(os.path.basename(path))[0]
    name = name.replace("OceanofPDF.com", "").replace("_", " ")
    return re.sub(r"\s+", " ", name).strip(" -")


def ocr_png(png_bytes: bytes) -> str:
    """Performs OCR on image bytes using pytesseract if available."""
    try:
        import pytesseract
        img = Image.open(io.BytesIO(png_bytes))
        return pytesseract.image_to_string(img, lang="eng")
    except Exception as e:
        logger.debug(f"OCR failed or tesseract not installed: {e}")
        return ""


class BookExtractor:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_single_book(self, path: str) -> List[Dict]:
        """
        Extracts all pages from a single PDF book, utilizing cache when available.
        """
        title = clean_book_title(path)
        cache_slug = re.sub(r"\W+", "_", title)[:60]
        cache_file = self.cache_dir / f"{cache_slug}.json"

        # Check cache
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    pages = json.load(f)
                    logger.info(f"Loaded from cache: '{title}' ({len(pages)} pages)")
                    return pages
            except Exception as e:
                logger.warning(f"Error reading cache for '{title}': {e}. Re-extracting...")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Book PDF not found: {path}")

        logger.info(f"Extracting native text from '{title}'...")
        reader = PdfReader(path)
        total = len(reader.pages)

        # Pass 1: Native text extraction
        texts = []
        for i in tqdm(range(total), desc=f"Text  | {title[:30]}"):
            try:
                page_text = reader.pages[i].extract_text() or ""
                texts.append(page_text)
            except (KeyError, IndexError, ValueError, OSError) as e:
                logger.debug(f"Failed extracting page {i}: {e}")
                texts.append("")

        # Pass 2: OCR for pages with insufficient native text
        need_ocr = [i for i, t in enumerate(texts) if len(t.strip()) < MIN_CHARS]
        if need_ocr:
            if not is_tesseract_available():
                logger.info(f"'{title}': {len(need_ocr)} of {total} pages have little native text, but Tesseract-OCR is not installed on PATH. Preserving extracted text.")
            else:
                logger.info(f"'{title}': {len(need_ocr)} of {total} pages need OCR")
                try:
                    doc = fitz.open(path)
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        for start in tqdm(range(0, len(need_ocr), BATCH), desc=f"OCR   | {title[:30]}"):
                            chunk = need_ocr[start:start + BATCH]
                            pngs = [doc[idx].get_pixmap(dpi=OCR_DPI).tobytes("png") for idx in chunk]
                            for idx, text in zip(chunk, pool.map(ocr_png, pngs)):
                                if text and len(text.strip()) > len(texts[idx].strip()):
                                    texts[idx] = text
                    doc.close()
                except Exception as e:
                    logger.warning(f"OCR skipped for '{title}' (Failed: {e})")

        pages = [
            {
                "book": title,
                "page": i + 1,
                "text": t,
                "source_type": "book",
                "file_path": path
            }
            for i, t in enumerate(texts) if t.strip()
        ]

        # Write to cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False)

        logger.info(f"Extraction complete for '{title}'. Cached {len(pages)} pages.")
        return pages

    def extract_all_books(self, custom_paths: Optional[List[str]] = None) -> List[Dict]:
        """
        Extracts all discovered PDF books.
        """
        paths = custom_paths or get_book_paths()
        if not paths:
            logger.warning("No PDF books found in data/books or workspace root.")
            return []

        all_pages = []
        for path in paths:
            try:
                pages = self.extract_single_book(path)
                all_pages.extend(pages)
            except Exception as e:
                logger.error(f"Error extracting book '{path}': {e}")

        logger.info(f"Loaded total of {len(all_pages)} pages from {len(paths)} book(s).")
        return all_pages


if __name__ == "__main__":
    extractor = BookExtractor()
    pages = extractor.extract_all_books()
    print(f"Total extracted pages: {len(pages)}")
