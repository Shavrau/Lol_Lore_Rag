"""
End-to-end Verification Test for Runeterra Lore RAG.
Tests ingestion of all books and web lore, query expansion, retrieval, and answering with Gemini.
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.book_extractor import BookExtractor
from src.rag_engine import RuneterraRAGEngine

def main():
    print("=" * 70)
    print("=== INICIANDO TESTE COMPLETO DO RUNETERRA LORE RAG ===")
    print("=" * 70)

    # 1. Carregando Livros e Lore Web
    print("\n[1/4] Extraindo / Carregando Livros...")
    book_extractor = BookExtractor()
    pages = book_extractor.extract_all_books()
    print(f"-> Total de paginas de livros carregadas: {len(pages)}")

    # 2. Inicializando Motor RAG
    print("\n[2/4] Inicializando Motor RAG e Ingerindo Fontes...")
    engine = RuneterraRAGEngine()
    engine.ingest(pages=pages, include_web_lore=True)
    print(f"-> Total de chunks gerados: {len(engine.chunks)}")
    print(f"-> Total de fontes distintas: {len(engine.source_ids)}")

    # 3. Teste de Pergunta sobre o Livro do Garen (First Shield)
    print("\n[3/4] Testando Pergunta 1 (Livro: Garen: First Shield)...")
    q1 = "Qual e a missao de Garen e quem o acompanha na Vanguarda Destemida em First Shield?"
    ans1, src1 = engine.answer_question(q1, top_k=4, mode="bm25")
    print(f"\nPERGUNTA: {q1}")
    print(f"\nRESPOSTA:\n{ans1}")
    print("\nFONTES CITADAS:")
    for s in src1:
        print(f" - {s[0]} | Paginas/Tipo: {s[1]} | Secao: {s[2]}")

    # 4. Teste de Pergunta sobre Ruination & Lore Web
    print("\n[4/4] Testando Pergunta 2 (Ruination & Viego)...")
    q2 = "Como Viego causou a Ruina e quem era Isolde?"
    ans2, src2 = engine.answer_question(q2, top_k=4, mode="bm25")
    print(f"\nPERGUNTA: {q2}")
    print(f"\nRESPOSTA:\n{ans2}")
    print("\nFONTES CITADAS:")
    for s in src2:
        print(f" - {s[0]} | Paginas/Tipo: {s[1]} | Secao: {s[2]}")

    print("\n" + "=" * 70)
    print("=== TESTE CONCLUIDO COM SUCESSO! ===")
    print("=" * 70)

if __name__ == "__main__":
    main()
