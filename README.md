# ⚔️ Runeterra Lore RAG (League of Legends)

Sistema de **Retrieval-Augmented Generation (RAG)** de alta precisão especializado no universo e lore de **League of Legends / Runeterra**.

Este projeto integra tanto **livros canônicos oficiais em PDF** (*Realms of Runeterra*, *Ruination*, contos e novelas) quanto **conteúdo online extraído da Web** (biografias oficiais via Riot Data Dragon, contos do Universo e artigos canônicos via MediaWiki API da League Wiki).

---

## 🚀 Novidades desta Versão (v2.0 & Web Expansion)

1. **Expansão de Dados (Web Lore)**:
   - Extrator integrado para extrair biografias e habilidades de **todos os 170+ campeões** (via Riot Data Dragon em português ou inglês).
   - Extrator de **Lore de Regiões** (Demacia, Noxus, Ionia, Ilhas das Sombras, Freljord, Shurima, etc.).
   - Extrator de **Contos e Histórias do Universo** arquivados pela League Wiki.
2. **Caminhos Dinâmicos e Portabilidade**:
   - Detecção automática de ambiente: funciona tanto no **Windows/Linux local** quanto no **Google Colab**.
   - Resolução flexível de pastas (`data/books`, `data/web_lore`, `data/cache`) sem quebrar referências.
3. **Ambiente com `.env`**:
   - Gerenciamento de chaves e hiperparâmetros centralizado via `python-dotenv`.
4. **Recuperação Híbrida Avançada (Hybrid Retrieval)**:
   - **Busca Esparsa (BM25)** com tokenização inteligente e remoção de stopwords.
   - **Busca Densa (BGE-M3)** multilíngue com cache de vetores (`.npy`).
   - **Fusão Recíproca de Ranks (RRF - Reciprocal Rank Fusion)** para combinar o melhor dos dois mundos.
   - **Reranker Cross-Encoder (BGE-Reranker-v2-m3)** para ordenar os candidatos mais relevantes.
   - **Neighbor Expansion**: inclusão de blocos adjacentes para não perder contexto contínuo.
5. **Geração Aterrada com Google Gemini**:
   - Expansão de consulta multilíngue e respostas estritamente aterradas nas fontes, com citação de livro/página ou documento web.

---

## 📁 Estrutura do Projeto

```text
c:\Rag_lol\
├── .agents/
│   └── skills/
│       └── runeterra-lore-rag/
│           └── SKILL.md            # Skill do Antigravity para gerenciar e consultar a lore
├── context/
│   ├── ARCHITECTURE.md             # Arquitetura detalhada do RAG
│   ├── LORE_SOURCES.md             # Descrição detalhada de fontes e canonicidade
│   └── RETRIEVAL_AND_EVAL.md       # Explicação técnica de BM25, BGE-M3 e Reranker
├── data/
│   ├── books/                      # PDFs dos livros oficiais (Realms of Runeterra, Ruination)
│   ├── web_lore/                   # Documentos extraídos da web
│   │   ├── champions/              # Biografias de campeões (.json e .md)
│   │   ├── regions/                # História e cultura das regiões (.json e .md)
│   │   └── stories/                # Contos e crônicas oficiais (.json e .md)
│   └── cache/                      # Cache de OCR e embeddings (.npy)
├── src/
│   ├── config.py                   # Configuração e variáveis de ambiente
│   ├── lore_extractor.py           # Extrator de Lore da Web (Data Dragon & Wiki API)
│   ├── book_extractor.py           # Extrator de PDFs e OCR
│   ├── rag_engine.py               # Motor RAG híbrido e integração com Gemini
│   └── app.py                      # Interface Web Gradio standalone
├── .env.example                    # Modelo de variáveis de ambiente
├── .env                            # Suas variáveis locais (API Key)
├── requirements.txt                # Dependências Python
├── pyproject.toml                  # Configuração para uv/pip
└── runeterra_rag_v2.ipynb          # Notebook Jupyter atualizado (compatível Local + Colab)
```

---

## ⚙️ Pré-requisitos e Instalação

### Usando `uv` (Recomendado):
```powershell
# Cria o ambiente virtual e instala as dependências
uv sync
```
Ou com `pip` padrão:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🔑 Configuração do `.env`

Copie o `.env.example` para `.env` (já criado) e configure sua chave da API do Google Gemini:

```env
GEMINI_API_KEY=sua_chave_aqui
GEMINI_MODEL=gemini-2.5-flash
LORE_LANG=pt_BR
```
Obtenha sua chave gratuitamente no [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## 🌐 Como Extrair Lore da Web

O módulo `src/lore_extractor.py` baixa e estrutura o conhecimento do universo de League of Legends:

### 1. Extrair biografias dos Campeões (Riot Data Dragon):
```powershell
# Extrai 20 campeões em português:
uv run python src/lore_extractor.py --champions --limit 20 --lang pt_BR

# Extrair TODOS os 170+ campeões:
uv run python src/lore_extractor.py --champions --lang pt_BR

# Extrair um campeão específico (ex: Viego ou Jinx):
uv run python src/lore_extractor.py --champion Viego --lang pt_BR
```

### 2. Extrair Lore das Regiões (League Wiki):
```powershell
uv run python src/lore_extractor.py --regions
```

### 3. Extrair Contos do Universo (League Wiki):
```powershell
# Extrai até 5 contos por região:
uv run python src/lore_extractor.py --stories --limit-stories 5
```

### 4. Extração Completa:
```powershell
uv run python src/lore_extractor.py --all
```

---

## 📖 Como Usar o RAG

### Opção 1: Via Notebook Jupyter (`runeterra_rag_v2.ipynb`)
Abra o notebook no VS Code / Cursor / Jupyter ou importe para o Google Colab.
O notebook agora detecta automaticamente se está no ambiente local ou no Colab, localiza os livros na pasta `data/books/` e ingere os arquivos da web de `data/web_lore/`.

### Opção 2: Interface Web Gradio (`src/app.py`)
Inicie a aplicação com:
```powershell
uv run python src/app.py
```
Acesse a URL indicada (ex: `http://127.0.0.1:7860`) no navegador para fazer perguntas com histórico e citações detalhadas!
