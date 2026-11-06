# ⚔️ Runeterra Lore RAG (League of Legends)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LLM](https://img.shields.io/badge/Google%20Gemini-Flash%202.5-orange.svg)](https://aistudio.google.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sistema de **Retrieval-Augmented Generation (RAG)** de alta precisão focado na **Lore Canônica de Runeterra e League of Legends**. 

O projeto une o conhecimento de **livros oficiais em PDF** (*Realms of Runeterra*, *Ruination*, contos e novelas) com **conteúdo canônico extraído da Web** (biografias oficiais via Riot Data Dragon e crônicas do Universo arquivadas pela League Wiki).

A aplicação opera sob estrita separação entre **narrativa ficcional canônica** e **mecânicas de jogo**. Perguntas sobre o destino de reinos, conflitos políticos e origens de campeões são respondidas com fatos canônicos e citações diretas das fontes, sem alucinações sobre tempos de recarga, itens ou dados competitivos.

---

## ✨ Principais Funcionalidades

* **Interface Web Hextech Moderna:** Layout imersivo em tons de obsidiana e ouro rúnico, com barra de navegação retrátil, histórico de fontes e citações clicáveis.
* **Plano de Fundo Dinâmico por Região:** O papel de parede transiciona suavemente em cross-fade de acordo com o campeão ou região consultada (**Demacia**, **Noxus**, **Ionia**, **Freljord**, **Shurima**, **Ilhas das Sombras**, **Piltover**, **Zaun**, **Águas de Sentina**, **Monte Targon**, **Bandle City**, **O Vazio** e **Ixtal**).
* **Citações e Fontes Canônicas:** Todas as respostas trazem chips com referências diretas (`[S1]`, `[S2]`) que abrem um painel interativo exibindo o trecho exato, páginas do livro e capítulo.
* **Pipeline RAG Híbrido (BM25 + BGE-M3 + Reranker):**
  * Busca Esparsa Lexical (BM25 com tokenização em português e expansão de sinônimos/termos de Runeterra).
  * Busca Vetorial Densa (BGE-M3 multilíngue).
  * Reciprocal Rank Fusion (RRF) combinando ranks densos e esparsos.
  * Cross-Encoder Reranker para garantir que apenas os trechos mais relevantes alimentem o contexto.
* **Geração Factual com Google Gemini:** Instruções rígidas para aterramento factual, impedindo invenções narrativas.

---

## 📁 Estrutura do Projeto

```text
Lol_Lore_Rag/
├── data/
│   ├── books/                      # PDFs oficiais (Realms of Runeterra, Ruination, Garen)
│   ├── web_lore/                   # Base canônica extraída da Web
│   │   ├── champions/              # Biografias de 170+ campeões (.json e .md)
│   │   ├── regions/                # História e cultura das 13 facções (.json e .md)
│   │   └── stories/                # Contos e crônicas oficiais do Universo (.json e .md)
│   └── cache/                      # Textos extraídos e pré-processados (.json)
├── src/
│   ├── config.py                   # Configurações de caminhos, modelo e variáveis de ambiente
│   ├── lore_extractor.py           # Web scraper para Riot Data Dragon e League Wiki API
│   ├── book_extractor.py           # Leitor de PDFs e parser estruturado
│   ├── rag_engine.py               # Motor RAG híbrido, BM25, RRF e integração Gemini
│   ├── server.py                   # Servidor FastAPI e rotas REST da aplicação web
│   └── app.py                      # Interface alternativa em Gradio
├── static/
│   ├── index.html                  # Interface completa (HTML5, CSS3, JS nativo)
│   └── img/
│       └── regions/                # Ilustrações canônicas em 2048px para cada facção
├── runeterra_rag_v2.ipynb          # Notebook de estudos (compatível Local e Google Colab)
├── start_local.bat                 # Inicializador rápido de 1 clique para Windows
├── start_public_app.bat            # Inicializador com túnel público Cloudflare
├── Dockerfile                      # Imagem de contêiner para deploy em nuvem
├── docker-compose.yml              # Orquestração local via Docker
├── pyproject.toml                  # Metadados do projeto e dependências para uv
├── requirements.txt                # Dependências para pip padrão
└── .env.example                    # Modelo para configuração de chaves
```

---

## 🚀 Como Rodar Localmente

### 1. Clonar o Repositório

```bash
git clone https://github.com/Shavrau/Lol_Lore_Rag.git
cd Lol_Lore_Rag
```

### 2. Configurar a Chave da API (Gemini)

Copie o arquivo `.env.example` para `.env`:

```bash
cp .env.example .env
```

Abra o arquivo `.env` e insira sua chave da API do Google Gemini:

```env
GEMINI_API_KEY=sua_chave_aqui
GEMINI_MODEL=gemini-2.5-flash
LORE_LANG=pt_BR
```

> Obtenha sua chave gratuita em [Google AI Studio](https://aistudio.google.com/app/apikey).

---

### 3. Instalar Dependências

#### Opção A: Usando `uv` (Recomendado, ultra rápido)
```bash
uv sync
```

#### Opção B: Usando `pip` padrão
```bash
python -m venv .venv

# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

### 4. Iniciar a Aplicação Web

No Windows, você pode simplesmente dar **dois cliques em `start_local.bat`**, ou executar via terminal:

```bash
uv run python src/server.py
```
*(ou `python src/server.py` caso use o venv com pip)*

Acesse no seu navegador:
👉 **`http://127.0.0.1:8080`**

---

## 🌐 Extração e Atualização de Lore da Web

Caso deseje atualizar as biografias ou baixar novas histórias do Universo:

```bash
# Extrair todas as biografias de campeões em português:
uv run python src/lore_extractor.py --champions --lang pt_BR

# Extrair um campeão específico (ex: Viego ou Jinx):
uv run python src/lore_extractor.py --champion Viego --lang pt_BR

# Extrair a lore detalhada das regiões:
uv run python src/lore_extractor.py --regions

# Extrair contos oficiais arquivados:
uv run python src/lore_extractor.py --stories --limit-stories 10

# Executar extração completa de uma vez:
uv run python src/lore_extractor.py --all
```

---

## 🐳 Executando com Docker

Se preferir rodar isolado em contêiner:

```bash
# Construir e rodar com docker-compose:
docker compose up --build

# Ou construir a imagem Docker diretamente:
docker build -t lol-lore-rag .
docker run -p 8080:8080 --env-file .env lol-lore-rag
```

Acesse em `http://localhost:8080`.

---

## ⚖️ Licença e Aviso Legal

Distribuído sob a licença [MIT](LICENSE).

*Aviso de Isenção: Este é um projeto independente de código aberto desenvolvido para fins educacionais e de pesquisa em Inteligência Artificial. Não é patrocinado, endossado ou afiliado à Riot Games, Inc. League of Legends e Runeterra são marcas registradas da Riot Games, Inc.*
