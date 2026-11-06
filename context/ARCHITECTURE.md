# 🏛️ Arquitetura do Sistema Runeterra Lore RAG

Este documento detalha o funcionamento arquitetural do sistema de RAG (Retrieval-Augmented Generation) para League of Legends, combinando fontes estáticas (livros em PDF) e fontes dinâmicas extraídas da Web.

---

## 📐 Visão Geral da Arquitetura

```mermaid
flowchart TD
    subgraph Ingestao ["1. Ingestão de Dados"]
        PDF["Livros PDF (Realms of Runeterra, Ruination)"] --> PDFExtract["Book Extractor (PyMuPDF + OCR Tesseract)"]
        WebData["Data Dragon & League Wiki API"] --> WebExtract["Web Lore Extractor (Champions, Regions, Stories)"]
    end

    subgraph Chunking ["2. Chunking e Normalização"]
        PDFExtract --> WordChunk["Chunking por Janela Deslizante (320 palavras, 50 overlap)"]
        WebExtract --> WebChunk["Chunking Estruturado com Metadados"]
        WordChunk --> ChunksStore["Catálogo de Chunks Unificado"]
        WebChunk --> ChunksStore
    end

    subgraph Indexacao ["3. Índices de Recuperação"]
        ChunksStore --> BM25["Índice Esparso BM25 (Okapi)"]
        ChunksStore --> DenseEmb["Embeddings Densos BAAI/bge-m3 (Cache .npy)"]
    end

    subgraph Consulta ["4. Fluxo de Consulta & Hybrid Search"]
        UserQ["Pergunta do Usuário (PT ou EN)"] --> Expander["Expansão de Query com Gemini (8-12 keywords)"]
        UserQ --> DenseSearch["Busca Densa (Similaridade Cosseno)"]
        Expander --> BM25Search["Busca BM25 (Boosting de Termos)"]
        DenseSearch & BM25Search --> RRF["Fusão RRF (Reciprocal Rank Fusion k=60)"]
        RRF --> Rerank["Reranker Cross-Encoder BAAI/bge-reranker-v2-m3"]
        Rerank --> Neighbor["Expansão de Vizinhos (Chunks Adjacentes)"]
    end

    subgraph Geracao ["5. Resposta Aterrada"]
        Neighbor --> GroundedContext["Contexto com Citações [S1, S2...]"]
        GroundedContext & UserQ --> GeminiLLM["Google Gemini (gemini-2.5-flash)"]
        GeminiLLM --> FinalAnswer["Resposta Rica com Tabela de Fontes"]
    end
```

---

## 🧩 Componentes do Sistema

### 1. Ingestão Multimodal de Documentos
- **PDFs de Livros**: O `src/book_extractor.py` extrai texto nativo via `pypdf`. Páginas com pouco texto passam por OCR de alta fidelidade (200 DPI) via `fitz` (PyMuPDF) e `pytesseract`.
- **Web Lore**: O `src/lore_extractor.py` consome a API do Riot Data Dragon (champions, spells, quotes) e MediaWiki API (histórias, regiões, contos canônicos), salvando arquivos `.json` e `.md` em `data/web_lore/`.

### 2. Chunking com Preservação de Fronteiras
- Janela de **320 palavras** com **50 palavras de sobreposição** (~420 tokens de prosa).
- Transcende quebras de página para manter cenas e diálogos intactos.
- Rastreia metadados essenciais: `source_name`, `page_start`, `page_end`, `chapter`, `category`, `region`.

### 3. Recuperação Híbrida (Hybrid Retrieval)
- **BM25**: Recuperação lexical eficiente para nomes próprios exatos (ex: *Camavor*, *Sanctum*, *Viego*, *Isolde*, *Petricite*).
- **BGE-M3**: Modelo de embedding denso multilíngue de 1024 dimensões com normalização L2.
- **RRF (Reciprocal Rank Fusion)**: Combina rankings de forma robusta e invariante a escalas:
  $$\text{RRF\_Score}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$
- **Reranker Cross-Encoder (`bge-reranker-v2-m3`)**: Avalia a relação entre pergunta e passagem completa com atenção cruzada total.

### 4. Expansão de Vizinhos (Neighbor Chunking)
- Para os melhores candidatos do reranking, o motor busca automaticamente o chunk anterior e posterior pertencentes ao mesmo documento/livro, costurando o texto e eliminando palavras duplicadas da sobreposição.

### 5. Geração e Citação
- O LLM recebe regras rígidas de ancoragem (grounding): nunca inventar informações e responder estritamente baseado no contexto.
- Retorna tabela estruturada com Fonte, Páginas e Capítulos utilizados.
