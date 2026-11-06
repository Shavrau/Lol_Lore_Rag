"""
FastAPI Server for Runeterra Lore RAG.
Provides a modern, high-performance Hextech web interface and REST API.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from src.book_extractor import BookExtractor
from src.config import BASE_DIR, DEFAULT_RETRIEVAL_MODE, GEMINI_MODEL
from src.rag_engine import RuneterraRAGEngine

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RuneterraServer")

app = FastAPI(title="Runeterra Lore RAG", version="2.0.0")

# Static files directory
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Shared Engine Instance
engine = RuneterraRAGEngine()
book_extractor = BookExtractor()

@app.on_event("startup")
def startup_event():
    logger.info("Initializing Runeterra Lore RAG Engine...")
    pages = book_extractor.extract_all_books()
    engine.ingest(pages=pages, include_web_lore=True, build_dense=False)
    logger.info(f"Engine ready: {len(engine.chunks)} chunks across {len(engine.source_ids)} sources.")

class QueryRequest(BaseModel):
    question: str
    selected_sources: Optional[List[str]] = None
    top_k: int = 4
    mode: str = DEFAULT_RETRIEVAL_MODE

class ExcerptItem(BaseModel):
    tag: str
    source: str
    type: str
    pages: str
    chapter: str
    text: str
    is_used: bool

class QueryResponse(BaseModel):
    answer: str
    sources: List[List[str]]
    excerpts: List[ExcerptItem] = []
    question: str
    region: str = "default"

# Regional Lore and Champion Mapping
CHAMPION_REGION_MAP = {
    # Demacia
    "garen": "demacia", "lux": "demacia", "jarvan": "demacia", "jarvan iv": "demacia",
    "shyvana": "demacia", "fiora": "demacia", "vayne": "demacia", "galio": "demacia",
    "poppy": "demacia", "lucian": "demacia", "sylas": "demacia", "xin zhao": "demacia",
    "sona": "demacia", "kayle": "demacia", "morgana": "demacia", "quinn": "demacia",
    # Noxus
    "darius": "noxus", "swain": "noxus", "draven": "noxus", "katarina": "noxus",
    "cassiopeia": "noxus", "talon": "noxus", "sion": "noxus", "samira": "noxus",
    "riven": "noxus", "mordekaiser": "noxus", "kled": "noxus", "briar": "noxus",
    "leblanc": "noxus", "vladimir": "noxus", "ambessa": "noxus",
    # Ionia
    "yasuo": "ionia", "yone": "ionia", "ahri": "ionia", "akali": "ionia",
    "irelia": "ionia", "karma": "ionia", "lee sin": "ionia", "shen": "ionia",
    "zed": "ionia", "kennen": "ionia", "master yi": "ionia", "wuju": "ionia",
    "sett": "ionia", "syndra": "ionia", "varus": "ionia", "jhin": "ionia",
    "kayn": "ionia", "lillia": "ionia", "xayah": "ionia", "rakan": "ionia",
    "ivern": "ionia", "wukong": "ionia", "hwei": "ionia",
    # Freljord
    "ashe": "freljord", "sejuani": "freljord", "lissandra": "freljord",
    "braum": "freljord", "anivia": "freljord", "volibear": "freljord",
    "ornn": "freljord", "olaf": "freljord", "tryndamere": "freljord",
    "trundle": "freljord", "nunu": "freljord", "willump": "freljord",
    "udyr": "freljord", "gragas": "freljord",
    # Shurima
    "azir": "shurima", "nasus": "shurima", "renekton": "shurima", "xerath": "shurima",
    "sivir": "shurima", "amumu": "shurima", "taliyah": "shurima", "k'sante": "shurima",
    "akshan": "shurima", "rammus": "shurima", "naafiri": "shurima",
    # Shadow Isles
    "viego": "shadow_isles", "thresh": "shadow_isles", "hecarim": "shadow_isles",
    "kalista": "shadow_isles", "karthus": "shadow_isles", "gwen": "shadow_isles",
    "yorick": "shadow_isles", "elise": "shadow_isles", "maokai": "shadow_isles",
    "vex": "shadow_isles", "senna": "shadow_isles", "isolde": "shadow_isles",
    "camavor": "shadow_isles", "helia": "shadow_isles",
    # Piltover
    "caitlyn": "piltover", "vi": "piltover", "jayce": "piltover", "ezreal": "piltover",
    "camille": "piltover", "heimerdinger": "piltover", "seraphine": "piltover",
    "orianna": "piltover", "corki": "piltover",
    # Zaun
    "jinx": "zaun", "ekko": "zaun", "warwick": "zaun", "singed": "zaun",
    "viktor": "zaun", "twitch": "zaun", "blitzcrank": "zaun", "dr. mundo": "zaun",
    "mundo": "zaun", "zac": "zaun", "zeri": "zaun", "renata": "zaun",
    "renata glasc": "zaun", "janna": "zaun", "urgot": "zaun",
    # Bilgewater
    "miss fortune": "bilgewater", "gangplank": "bilgewater", "graves": "bilgewater",
    "twisted fate": "bilgewater", "illaoi": "bilgewater", "pyke": "bilgewater",
    "nautilus": "bilgewater", "nilah": "bilgewater", "tahm kench": "bilgewater", "fizz": "bilgewater",
    # Targon
    "pantheon": "targon", "leona": "targon", "diana": "targon", "taric": "targon",
    "zoe": "targon", "aurelion sol": "targon", "asol": "targon", "soraka": "targon",
    "aphelios": "targon", "alune": "targon",
    # The Void
    "kai'sa": "the_void", "kaisa": "the_void", "kassadin": "the_void",
    "kha'zix": "the_void", "khazix": "the_void", "cho'gath": "the_void",
    "chogath": "the_void", "vel'koz": "the_void", "velkoz": "the_void",
    "kog'maw": "the_void", "kogmaw": "the_void", "rek'sai": "the_void",
    "reksai": "the_void", "bel'veth": "the_void", "belveth": "the_void",
    "malzahar": "the_void", "icathia": "the_void", "void": "the_void",
    # Bandle City
    "teemo": "bandle_city", "tristana": "bandle_city", "lulu": "bandle_city",
    "veigar": "bandle_city", "rumble": "bandle_city", "yuumi": "bandle_city", "gnar": "bandle_city",
    # Ixtal
    "qiyana": "ixtal", "neeko": "ixtal", "nidalee": "ixtal", "rengar": "ixtal",
    "zyra": "ixtal", "milio": "ixtal", "malphite": "ixtal",
}

REGION_KEYWORDS = {
    "demacia": "demacia",
    "noxus": "noxus",
    "ionia": "ionia",
    "freljord": "freljord",
    "shurima": "shurima",
    "sombras": "shadow_isles",
    "shadow isles": "shadow_isles",
    "ruina": "shadow_isles",
    "camavor": "shadow_isles",
    "helia": "shadow_isles",
    "piltover": "piltover",
    "zaun": "zaun",
    "bilgewater": "bilgewater",
    "sentina": "bilgewater",
    "targon": "targon",
    "vazio": "the_void",
    "void": "the_void",
    "icathia": "the_void",
    "bandle": "bandle_city",
    "bandopolis": "bandle_city",
    "ixtal": "ixtal",
}

def detect_region(question: str, sources: List[List[str]] = None, excerpts: List[ExcerptItem] = None) -> str:
    text = (question or "").lower()
    
    # 1. Direct region keywords in question
    for kw, reg in REGION_KEYWORDS.items():
        if kw in text:
            return reg
            
    # 2. Champion name in question
    for champ, reg in CHAMPION_REGION_MAP.items():
        if champ in text:
            return reg
            
    # 3. Check retrieved sources
    if sources:
        for s in sources:
            s_name = (s[0] if isinstance(s, list) and s else str(s)).lower()
            for kw, reg in REGION_KEYWORDS.items():
                if kw in s_name:
                    return reg
            for champ, reg in CHAMPION_REGION_MAP.items():
                if champ in s_name:
                    return reg
                    
    if excerpts:
        for ex in excerpts:
            ex_source = ex.source.lower() if hasattr(ex, "source") else ""
            for kw, reg in REGION_KEYWORDS.items():
                if kw in ex_source:
                    return reg
            for champ, reg in CHAMPION_REGION_MAP.items():
                if champ in ex_source:
                    return reg

    return "default"

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "model": GEMINI_MODEL,
        "default_mode": DEFAULT_RETRIEVAL_MODE,
        "total_chunks": len(engine.chunks),
        "total_sources": len(engine.source_ids),
        "sources": sorted(list(engine.source_ids.keys())),
    }

@app.post("/api/query", response_model=QueryResponse)
def query_oracle(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")

    sources = req.selected_sources if req.selected_sources and len(req.selected_sources) > 0 else None
    try:
        answer, sources_cited, excerpts = engine.answer_question(
            question=req.question,
            selected_sources=sources,
            top_k=req.top_k,
            mode=req.mode or DEFAULT_RETRIEVAL_MODE,
            return_excerpts=True,
        )
        detected_region = detect_region(req.question, sources_cited, excerpts)
        return QueryResponse(
            answer=answer,
            sources=sources_cited,
            excerpts=excerpts,
            question=req.question,
            region=detected_region,
        )
    except Exception as e:
        logger.error(f"Error querying oracle: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Runeterra Lore RAG</h1><p>index.html not found</p>")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("src.server:app", host=host, port=port, reload=False)
