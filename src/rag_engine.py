"""
High-Performance Hybrid RAG Engine for League of Legends Lore.
Combines Book Pages (PDFs) and Web Lore (Data Dragon & Wiki Stories).
Features BM25 + BGE-M3 Dense Search + RRF Fusion + Cross-Encoder Reranking + Gemini QA.
"""

import hashlib
import json
import logging
import math
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.config import (
    CACHE_DIR,
    CANDIDATES,
    CHUNK_WORDS,
    DEFAULT_RETRIEVAL_MODE,
    DEVICE,
    EMBED_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    NEIGHBORS_FOR_TOP_N,
    OVERLAP_WORDS,
    RERANK_MODEL,
    RERANK_POOL,
    RRF_K,
    WEB_LORE_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAGEngine")

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "was", "were", "are", "his",
    "her", "their", "its", "has", "have", "had", "who", "what", "which", "when",
    "where", "how", "why", "does", "did", "about", "into", "than", "then", "there",
    "but", "not", "you", "your", "she", "him", "they", "them", "been", "will", "would",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", "por", "para", "com",
    "um", "uma", "uns", "umas", "ele", "ela", "eles", "elas", "que", "como", "onde", "quando"
}

HEADING_RE = re.compile(
    r"^\s*(?:CHAPTER|PART|PROLOGUE|EPILOGUE|INTERLUDE)\b.{0,60}$"
    r"|^\s*(?:Chapter|Part)\s+(?:\d+|[IVXLC]+)\b.{0,40}$"
    r"|^\s*#{1,3}\s+(.+)$"
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", normalize(text))
    return [t[:6] if len(t) > 6 else t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def clean_page_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = [ln for ln in text.splitlines() if not re.fullmatch(r"\s*\d{1,4}\s*", ln)]
    return "\n".join(lines)


# =============================================================================
# BM25 Sparse Index
# =============================================================================
class BM25Index:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(c.values()) for c in tfs]
        self.avg_len = (sum(self.lens) / len(self.lens)) if self.lens else 1
        self.postings = defaultdict(list)
        for i, c in enumerate(tfs):
            for t, f in c.items():
                self.postings[t].append((i, f))
        n = len(docs)
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        }

    def search(self, query: str, top_k: int = 50, allowed: Optional[Set[int]] = None) -> List[Tuple[int, float]]:
        scores = defaultdict(float)
        for term, qtf in Counter(tokenize(query)).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, f in self.postings[term]:
                if allowed is not None and doc_id not in allowed:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[doc_id] / self.avg_len)
                scores[doc_id] += qtf * idf * f * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# =============================================================================
# Runeterra RAG Engine
# =============================================================================
class RuneterraRAGEngine:
    def __init__(
        self,
        chunk_words: int = CHUNK_WORDS,
        overlap_words: int = OVERLAP_WORDS,
        embed_model: str = EMBED_MODEL,
        rerank_model: str = RERANK_MODEL,
    ):
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words
        self.embed_model_name = embed_model
        self.rerank_model_name = rerank_model

        self.chunks: List[Dict] = []
        self.source_ids: Dict[str, Set[int]] = defaultdict(set)
        self.bm25: Optional[BM25Index] = None
        self.embeddings: Optional[np.ndarray] = None

        self._embedder = None
        self._reranker = None
        self._genai_client = None

    # Lazy-loaded models
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedder '{self.embed_model_name}' on {DEVICE}...")
            self._embedder = SentenceTransformer(self.embed_model_name, device=DEVICE)
            self._embedder.max_seq_length = 512
            if DEVICE == "cuda":
                self._embedder.half()
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker '{self.rerank_model_name}' on {DEVICE}...")
            self._reranker = CrossEncoder(self.rerank_model_name, device=DEVICE, max_length=512)
        return self._reranker

    @property
    def genai_client(self):
        if self._genai_client is None:
            if not GEMINI_API_KEY:
                logger.warning("GEMINI_API_KEY not configured. Set it in .env to enable answering.")
                return None
            from google import genai
            self._genai_client = genai.Client(api_key=GEMINI_API_KEY)
        return self._genai_client

    # =========================================================================
    # Chunking & Document Processing
    # =========================================================================
    def build_chunks_from_pages(self, pages: List[Dict]) -> List[Dict]:
        """Processes book pages into continuous, overlapping chunks."""
        step = self.chunk_words - self.overlap_words
        book_chunks = []

        for book in sorted({p["book"] for p in pages}):
            book_pages = sorted((p for p in pages if p["book"] == book), key=lambda p: p["page"])
            words, word_page, page_chapter = [], [], {}
            chapter = ""
            for p in book_pages:
                text = clean_page_text(p["text"])
                for ln in text.splitlines():
                    if HEADING_RE.match(ln):
                        chapter = ln.strip("# \t")
                page_chapter[p["page"]] = chapter
                w = text.split()
                words.extend(w)
                word_page.extend([p["page"]] * len(w))

            for start in range(0, len(words), step):
                window = words[start:start + self.chunk_words]
                if start > 0 and len(window) <= self.overlap_words:
                    break
                first_page = word_page[start]
                last_page = word_page[start + len(window) - 1]
                book_chunks.append({
                    "id": len(self.chunks) + len(book_chunks),
                    "source_type": "book",
                    "source_name": book,
                    "book": book,
                    "text": " ".join(window),
                    "page_start": first_page,
                    "page_end": last_page,
                    "chapter": page_chapter[first_page],
                })
        return book_chunks

    def build_chunks_from_web_lore(self, web_lore_dir: Optional[Path] = None) -> List[Dict]:
        """Loads and chunks web lore documents (.json files)."""
        target_dir = web_lore_dir or WEB_LORE_DIR
        web_chunks = []
        if not target_dir.exists():
            return []

        json_files = list(target_dir.rglob("*.json"))
        logger.info(f"Scanning web lore: found {len(json_files)} document(s)...")

        step = self.chunk_words - self.overlap_words

        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading web lore file {jf}: {e}")
                continue

            text = doc.get("text", "")
            if not text.strip():
                continue

            category = doc.get("category", "web_lore")
            name = doc.get("name")
            raw_title = doc.get("title")
            if category == "champion_bio" and name:
                title = f"{name} - {raw_title}" if raw_title else name
            else:
                title = raw_title or name or jf.stem
            source_name = f"Web: {title}"
            region = doc.get("region", "")

            words = text.split()
            if len(words) <= self.chunk_words:
                web_chunks.append({
                    "id": len(self.chunks) + len(web_chunks),
                    "source_type": "web",
                    "source_name": source_name,
                    "category": category,
                    "region": region,
                    "title": title,
                    "chapter": f"Category: {category}",
                    "page_start": 1,
                    "page_end": 1,
                    "text": text,
                })
            else:
                for start in range(0, len(words), step):
                    window = words[start:start + self.chunk_words]
                    if start > 0 and len(window) <= self.overlap_words:
                        break
                    part_num = (start // step) + 1
                    web_chunks.append({
                        "id": len(self.chunks) + len(web_chunks),
                        "source_type": "web",
                        "source_name": source_name,
                        "category": category,
                        "region": region,
                        "title": title,
                        "chapter": f"{title} (Part {part_num})",
                        "page_start": part_num,
                        "page_end": part_num,
                        "text": " ".join(window),
                    })

        return web_chunks

    def ingest(self, pages: Optional[List[Dict]] = None, include_web_lore: bool = True, build_dense: bool = False):
        """Builds all chunks and initializes BM25 & Dense indices."""
        self.chunks = []
        self.source_ids = defaultdict(set)

        if pages:
            book_chunks = self.build_chunks_from_pages(pages)
            self.chunks.extend(book_chunks)

        if include_web_lore:
            web_chunks = self.build_chunks_from_web_lore()
            self.chunks.extend(web_chunks)

        # Re-index chunk IDs
        for i, c in enumerate(self.chunks):
            c["id"] = i
            self.source_ids[c["source_name"]].add(i)

        logger.info(f"Ingested {len(self.chunks)} total chunks across {len(self.source_ids)} source(s):")
        for src, ids in sorted(self.source_ids.items()):
            logger.info(f" - {src}: {len(ids)} chunks")

        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25 = BM25Index([c["text"] for c in self.chunks])

        # Build / Load embeddings if requested
        if build_dense:
            self.load_or_build_embeddings()

    # =========================================================================
    # Dense Embeddings & Cache
    # =========================================================================
    def embed_texts(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        prefix = ("query: " if is_query else "passage: ") if "e5" in self.embed_model_name.lower() else ""
        vecs = self.embedder.encode(
            [prefix + t for t in texts],
            batch_size=32 if DEVICE == "cuda" else 8,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def load_or_build_embeddings(self) -> np.ndarray:
        if not self.chunks:
            self.embeddings = np.zeros((0, 1024), dtype=np.float32)
            return self.embeddings

        h = hashlib.sha1(self.embed_model_name.encode())
        for c in self.chunks:
            h.update(c["text"].encode("utf-8"))
        cache_path = CACHE_DIR / f"emb_{h.hexdigest()[:16]}.npy"

        if cache_path.exists():
            logger.info(f"Loaded embeddings from cache: {cache_path.name}")
            self.embeddings = np.load(cache_path)
            return self.embeddings

        logger.info(f"Generating dense embeddings for {len(self.chunks)} chunks with {self.embed_model_name}...")
        self.embeddings = self.embed_texts([c["text"] for c in self.chunks])
        np.save(cache_path, self.embeddings)
        logger.info(f"Cached embeddings to {cache_path.name}")
        return self.embeddings

    def dense_search(self, query: str, top_k: int = 50, allowed: Optional[Set[int]] = None) -> List[Tuple[int, float]]:
        if self.embeddings is None or len(self.embeddings) == 0:
            self.load_or_build_embeddings()
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        q = self.embed_texts([query], is_query=True)[0]
        sims = self.embeddings @ q
        if allowed is not None:
            mask = np.full(len(sims), -np.inf, dtype=np.float32)
            mask[list(allowed)] = 0.0
            sims = sims + mask
        top = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in top if np.isfinite(sims[i])]

    # =========================================================================
    # Hybrid Retrieval & Reranking
    # =========================================================================
    @staticmethod
    def rrf_fuse(rankings: List[List[Tuple[int, float]]], k: int = RRF_K) -> List[Tuple[int, float]]:
        scores = defaultdict(float)
        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                scores[doc_id] += 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(
        self,
        question: str,
        keywords: str = "",
        sources: Optional[List[str]] = None,
        mode: str = DEFAULT_RETRIEVAL_MODE,
        top_k: int = 4,
    ) -> List[Tuple[int, float]]:
        allowed = set().union(*(self.source_ids[s] for s in sources)) if sources else None

        bm25_hits = self.bm25.search(f"{question} {keywords} {keywords}", CANDIDATES, allowed)
        if mode == "bm25":
            return bm25_hits[:top_k]

        dense_hits = self.dense_search(question, CANDIDATES, allowed)
        if mode == "dense":
            return dense_hits[:top_k]

        rankings = [bm25_hits, dense_hits]
        if keywords:
            rankings.append(self.dense_search(keywords, CANDIDATES, allowed))
        fused = self.rrf_fuse(rankings)
        if mode == "hybrid":
            return fused[:top_k]

        pool = fused[:RERANK_POOL]
        if not pool:
            return []
        pairs = [(question, self.chunks[cid]["text"]) for cid, _ in pool]
        scores = self.reranker.predict(pairs, batch_size=8, show_progress_bar=False)
        reranked = sorted(zip([cid for cid, _ in pool], map(float, scores)), key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def build_context_blocks(self, hit_ids: List[int], neighbors_for: int = NEIGHBORS_FOR_TOP_N) -> List[Dict]:
        selected = set(hit_ids)
        for cid in hit_ids[:neighbors_for]:
            for nid in (cid - 1, cid + 1):
                if 0 <= nid < len(self.chunks) and self.chunks[nid]["source_name"] == self.chunks[cid]["source_name"]:
                    selected.add(nid)

        runs, run = [], []
        for cid in sorted(selected):
            if run and cid == run[-1] + 1 and self.chunks[cid]["source_name"] == self.chunks[run[-1]]["source_name"]:
                run.append(cid)
            else:
                if run:
                    runs.append(run)
                run = [cid]
        if run:
            runs.append(run)

        blocks = []
        for run in runs:
            parts = [self.chunks[run[0]]["text"]]
            for cid in run[1:]:
                parts.append(" ".join(self.chunks[cid]["text"].split()[self.overlap_words:]))
            first, last = self.chunks[run[0]], self.chunks[run[-1]]
            blocks.append({
                "source_type": first.get("source_type", "book"),
                "source_name": first.get("source_name", "Unknown"),
                "chapter": first.get("chapter", ""),
                "page_start": first.get("page_start", 1),
                "page_end": last.get("page_end", 1),
                "text": " ".join(parts),
                "chunk_ids": run,
            })
        return blocks

    # =========================================================================
    # Gemini Generation & QA
    # =========================================================================
    def expand_query(self, question: str) -> str:
        if not self.genai_client:
            return ""
        prompt = f"""The user asks a question about League of Legends lore (Runeterra, champions, regions, artifacts, events).
Write 8 to 12 English search keywords, comma-separated, to find the answer in the lore documents.
Include proper nouns (characters, places, weapons, events) exactly as spelled in English.
Output ONLY the comma-separated keywords.

Question: {question}"""
        try:
            r = self.genai_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return (r.text or "").strip()
        except Exception as e:
            logger.debug(f"Query expansion failed: {e}")
            return ""

    def answer_question(
        self,
        question: str,
        selected_sources: Optional[List[str]] = None,
        top_k: int = 4,
        mode: str = DEFAULT_RETRIEVAL_MODE,
        return_excerpts: bool = False,
    ):
        question = (question or "").strip()
        if not question:
            empty_res = "Por favor, digite uma pergunta primeiro.", []
            return (*empty_res, []) if return_excerpts else empty_res

        sources = selected_sources or list(self.source_ids.keys())
        keywords = self.expand_query(question) if mode != "dense" else ""
        hits = self.retrieve(question, keywords, sources, mode, int(top_k))
        if not hits:
            not_found = "Nenhuma informação correspondente foi encontrada nos registros e arquivos oficiais de Runeterra.", []
            return (*not_found, []) if return_excerpts else not_found

        blocks = self.build_context_blocks([cid for cid, _ in hits])
        parts = []
        for n, b in enumerate(blocks, start=1):
            src_label = b["source_name"]
            chapter_info = f" | Seção/Capítulo: {b['chapter']}" if b.get("chapter") else ""
            page_info = f" | Páginas: {b['page_start']}-{b['page_end']}" if b.get("source_type") == "book" else ""
            parts.append(f"[S{n}] Fonte: {src_label}{chapter_info}{page_info}\n{b['text']}")
        context = "\n\n".join(parts)

        if not self.genai_client:
            no_key = "Erro: GEMINI_API_KEY não foi configurada no arquivo .env.", []
            return (*no_key, []) if return_excerpts else no_key

        prompt = f"""Você é o motor de consulta factual da Lore de Runeterra (League of Legends).
Regras obrigatórias:
1. Responda com base EXCLUSIVAMENTE nas fontes documentais fornecidas abaixo. Não invente ou presuma fatos.
2. Responda de forma direta, clara e factual em português.
3. Não use introduções pomposas, clichês de IA ou dramatizações teatrais. Vá direto aos fatos narrativos.
4. Ao citar informações específicas de um trecho, inclua a marcação da fonte correspondente (ex: [S1], [S2]) diretamente no texto.
5. Formate a resposta no padrão estrito:
ANSWER: <sua resposta factual, bem pontuada e citada>
USED: <ids de fontes utilizados, ex: S1,S2>
6. Se a resposta não constar no contexto, responda:
ANSWER: Nenhuma informação correspondente foi encontrada nos registros e arquivos oficiais de Runeterra.
USED: none

--- CONTEXTO ---
{context}

--- PERGUNTA ---
{question}
"""
        try:
            r = self.genai_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            raw = (r.text or "").strip()
        except Exception as e:
            logger.warning(f"Primary model generation failed ({e}), attempting fallback model...")
            try:
                fallback_model = "gemini-2.5-flash" if "flash-lite" in GEMINI_MODEL else "gemini-2.5-flash-lite"
                r = self.genai_client.models.generate_content(model=fallback_model, contents=prompt)
                raw = (r.text or "").strip()
            except Exception as e2:
                logger.error(f"Fallback generation also failed: {e2}")
                fallback_excerpts = [
                    {
                        "tag": f"S{n}",
                        "source": b["source_name"],
                        "type": b.get("source_type", "web"),
                        "pages": f"Páginas {b['page_start']}-{b['page_end']}" if b.get("source_type") == "book" else "Arquivo Web",
                        "chapter": b.get("chapter", ""),
                        "text": b["text"][:750].strip() + ("..." if len(b["text"]) > 750 else ""),
                        "is_used": True,
                    }
                    for n, b in enumerate(blocks, start=1)
                ]
                err_msg = f"As fontes oficiais e documentos de Runeterra foram consultados com sucesso, mas a síntese de texto encontrou uma indisponibilidade temporária na API: {e}."
                return (err_msg, [[b["source_name"], f"{b['page_start']}-{b['page_end']}" if b.get("source_type") == "book" else "Web", b.get("chapter", "-")] for b in blocks], fallback_excerpts) if return_excerpts else (err_msg, [])

        m_ans = re.search(r"ANSWER:\s*(.*?)\s*(?:USED:|$)", raw, re.S)
        m_used = re.search(r"USED:\s*(.*)", raw)
        answer = m_ans.group(1).strip() if m_ans else raw

        used_ids = set(re.findall(r"S(\d+)", m_used.group(1))) if m_used else set()
        sources_table = [
            [
                b["source_name"],
                f"{b['page_start']}-{b['page_end']}" if b.get("source_type") == "book" else "Web",
                b.get("chapter", "-")
            ]
            for n, b in enumerate(blocks, start=1)
            if str(n) in used_ids
        ]

        if return_excerpts:
            excerpts = [
                {
                    "tag": f"S{n}",
                    "source": b["source_name"],
                    "type": b.get("source_type", "web"),
                    "pages": f"Páginas {b['page_start']}-{b['page_end']}" if b.get("source_type") == "book" else "Arquivo Web",
                    "chapter": b.get("chapter", ""),
                    "text": b["text"][:750].strip() + ("..." if len(b["text"]) > 750 else ""),
                    "is_used": str(n) in used_ids,
                }
                for n, b in enumerate(blocks, start=1)
            ]
            return answer, sources_table, excerpts

        return answer, sources_table
