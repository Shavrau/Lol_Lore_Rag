# 🔍 Estratégia de Recuperação e Avaliação (Hybrid RAG)

Este guia explica os detalhes matemáticos e práticos do pipeline de recuperação e como avaliar a qualidade das respostas obtidas.

---

## 🎯 Por que Busca Híbrida?

Em lore de fantasia épica como Runeterra:
1. **Nomes próprios e termos arcanos** (*Camavor, Petricite, Darkin, Nagakabouros, Ruined King, Helia*):
   - Modelos densos (embeddings) frequentemente tratam nomes inventados ou específicos como palavras raras com baixa densidade vetorial.
   - O algoritmo **BM25** (esparso) se sobressai na recuperação exata desses termos através de sua frequência invertida (IDF).
2. **Consultas conceituais e semânticas** (*"como a loucura de amor destruiu o reino", "qual a natureza da magia curativa"*):
   - Modelos densos como o **BAAI/bge-m3** capturam analogias semânticas que o BM25 é incapaz de mapear por palavras-chave diretas.
3. **Cross-Lingual Matching (Português -> Inglês)**:
   - Muitos livros estão em inglês, enquanto o usuário faz perguntas em português.
   - O pipeline resolve isso gerando palavras-chave em inglês com o Gemini (`expand_query`) e usando modelos multilíngues (`bge-m3` e `bge-reranker-v2-m3`).

---

## 🔢 Fórmulas e Algoritmos

### 1. BM25 (Okapi BM25 com Inverted Index)
Para um documento $D$ e termo de consulta $q$:
$$\text{IDF}(q) = \ln\left(1 + \frac{N - n(q) + 0.5}{n(q) + 0.5}\right)$$
$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
Configurações adotadas: $k_1 = 1.5$, $b = 0.75$.

### 2. Reciprocal Rank Fusion (RRF)
Fusão dos candidatos recuperados pelo BM25 e pelos embeddings densos:
$$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + \text{rank}_m(d)}$$
Onde $k = 60$ suaviza a penalidade de documentos ligeiramente abaixo no topo do ranking.

### 3. Cross-Encoder Reranker (`bge-reranker-v2-m3`)
Diferente dos bi-encoders que geram vetores separados, o cross-encoder recebe simultaneamente o par `(pergunta, trecho)` no mecanismo de atenção de 512 tokens:
$$s = \text{Softmax}\left(\text{CrossEncoder}(Q, C)\right)$$
Ele reordena a piscina dos 20 candidatos fundidos e seleciona os Top-K mais precisos.

---

## 🔬 Modos de Comparação (`compare_modes`)

O motor permite comparar os 4 modos de busca para qualquer pergunta:
- `bm25`: Apenas busca lexical por frequência de termos.
- `dense`: Apenas similaridade vetorial por cosseno.
- `hybrid`: Fusão RRF de BM25 e BGE-M3.
- `hybrid+rerank`: Fusão RRF refinada com Cross-Encoder (Recomendado).
