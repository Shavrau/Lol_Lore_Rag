# 📚 Fontes de Lore de League of Legends & Runeterra

O universo ficcional de Runeterra é distribuído através de múltiplos formatos, desde livros físicos publicados pela Riot Games até conteúdo dinâmico no cliente e na web.

---

## 📖 1. Livros Oficiais (PDF / Físico)

| Título | Autor / Publicação | Foco Principal | Status no RAG |
| :--- | :--- | :--- | :--- |
| **Realms of Runeterra** | Riot Games (2019) | Enciclopédia canônica abrangendo as regiões (Demacia, Noxus, Ionia, Piltover, Zaun, Freljord, Targon, Shurima, Águas de Sentina, Ilhas das Sombras, Ixtal, Bandleópolis). Contém contos exclusivos e mapas. | Ingerido nativamente via OCR e PDF extractor (`data/books/`) |
| **Ruination** | Anthony Reynolds (2022) | Romance focado na queda de Camavor, ascensão e loucura do Rei Viego, Kalista, Hecarim, Thresh e a criação das Ilhas das Sombras. | Ingerido nativamente via OCR e PDF extractor (`data/books/`) |
| **Garen: First Shield** | Anthony Reynolds (2020) | Novela sobre a vanguarda destemida de Demacia além das fronteiras. | Suportado (adicionar em `data/books/`) |

---

## 🌐 2. Fontes da Web (Extração Automatizada)

O extrator `src/lore_extractor.py` coleta e atualiza as seguintes fontes online:

### A. Riot Data Dragon (API Oficial)
- **Endpoint**: `https://ddragon.leagueoflegends.com/cdn/{version}/data/{locale}/`
- **Conteúdo**:
  - Biografias oficiais canônicas de todos os mais de 170 campeões.
  - Lore das habilidades (passivas, Q, W, E, R), que revelam aspectos do estilo de combate e magia.
  - Frases e títulos dos campeões.
  - Suporte nativo aos idiomas `pt_BR` (Português Brasileiro) e `en_US` (Inglês).

### B. League of Legends Wiki (MediaWiki API)
- **Endpoint**: `https://leagueoflegends.fandom.com/api.php`
- **Conteúdo**:
  - **Artigos de Região**: Geografia, sociedade, conflitos históricos, cultura e criaturas mágicas.
  - **Contos do Universo**: Centenas de contos curtos (Color Stories) de campeões e eventos históricos (ex: *For Demacia*, *Silence for the Damned*, *Sisterhood of War*, *Twilight of the Gods*).
  - **Categorias exploradas**:
    - `Category:Lore of Demacia`
    - `Category:Lore of Noxus`
    - `Category:Lore of Ionia`
    - `Category:Lore of Freljord`
    - `Category:Lore of Shadow Isles`
    - `Category:Lore of Shurima`
    - `Category:Lore of Piltover`
    - `Category:Lore of Zaun`
    - `Category:Lore of Bilgewater`
    - `Category:Lore of Targon`
    - `Category:Lore of Bandle City`
    - `Category:Lore of Ixtal`
    - `Category:Lore of Runeterra`

---

## 🏷️ Metadados de Documento no RAG

Todo documento ingerido recebe metadados estruturados para filtragem e citação:
```json
{
  "id": "champ_viego_pt_br",
  "source_type": "web",
  "source_name": "Web: Viego - O Rei Destruído",
  "category": "champion_bio",
  "region": "Ilhas das Sombras / Camavor",
  "title": "Viego - O Rei Destruído",
  "text": "..."
}
```
Isso permite aos usuários selecionar na interface se desejam pesquisar apenas em livros específicos, apenas em biografias de campeões ou em todo o acervo.
