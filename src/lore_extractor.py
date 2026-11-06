"""
Web Lore Extractor for League of Legends (Runeterra Lore RAG).
Focuses strictly on narrative character lore, biographies, world history, and canonical Universe stories.
Does NOT extract or ingest in-game champion kit mechanics, abilities, spells, cooldowns, stats, or tips.
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup

from src.config import WEB_LORE_DIR, LORE_LANG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LoreExtractor")

DDRAGON_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
WIKI_API_URL = "https://leagueoflegends.fandom.com/api.php"
WIKI_HEADERS = {"User-Agent": "RuneterraLoreRAG/2.0 (Canonical Lore Archive; Educational Research)"}

REGIONS = [
    "Demacia",
    "Noxus",
    "Ionia",
    "Freljord",
    "Piltover",
    "Zaun",
    "Bilgewater",
    "Shurima",
    "Targon",
    "Shadow Isles",
    "Ixtal",
    "Bandle City",
    "The Void",
    "Runeterra",
]

# Major Lore-Rich Champions across all regions of Runeterra
MAJOR_LORE_CHAMPIONS = [
    # Camavor & Shadow Isles
    "Viego", "Kalista", "Hecarim", "Thresh", "Gwen", "Yorick", "Senna", "Lucian", "Vex",
    # Demacia
    "Garen", "Lux", "Sylas", "Jarvan IV", "Shyvana", "Galio", "Fiora", "Vayne", "Xin Zhao",
    # Noxus
    "Swain", "Darius", "LeBlanc", "Mordekaiser", "Sion", "Draven", "Katarina", "Vladimir", "Rell", "Ambessa",
    # Ionia
    "Yasuo", "Yone", "Riven", "Shen", "Zed", "Akali", "Karma", "Jhin", "Irelia", "Ahri", "Kayn", "Syndra",
    # Freljord
    "Ashe", "Sejuani", "Lissandra", "Volibear", "Ornn", "Anivia", "Braum", "Udyr", "Tryndamere",
    # Shurima & Darkin
    "Azir", "Xerath", "Renekton", "Nasus", "Taliyah", "Sivir", "Aatrox", "Varus", "Naafiri", "K'Sante",
    # Piltover & Zaun
    "Jinx", "Vi", "Caitlyn", "Jayce", "Viktor", "Warwick", "Singed", "Ekko", "Camille", "Heimerdinger",
    # Targon
    "Pantheon", "Diana", "Leona", "Taric", "Aurelion Sol", "Zoe", "Aphelios", "Soraka",
    # Bilgewater
    "Miss Fortune", "Gangplank", "Illaoi", "Twisted Fate", "Graves", "Pyke",
    # The Void
    "Bel'Veth", "Kai'Sa", "Kassadin", "Malzahar",
    # Wanderers & Mythic Runeterra
    "Ryze", "Jax", "Kindred", "Fiddlesticks", "Bard", "Tahm Kench"
]


def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be used safely as a filename."""
    return re.sub(r'[\\/*?:"<>| ]', "_", name).strip("_")


def clean_wikitext(text: str) -> str:
    """Cleans MediaWiki wikitext into crisp, readable markdown lore."""
    # Convert champion/item templates {{cbi|Kalista}} -> Kalista
    text = re.sub(r'\{\{(?:cbi|ci|ai|bi|ri|sbi)\|([^}|]+)(?:\|[^}]+)?\}\}', r'\1', text, flags=re.IGNORECASE)
    # Generic templates {{...}} -> empty
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    # Wikilinks [[Target|Display]] -> Display, [[Target]] -> Target
    text = re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', text)
    # Markdown formatting
    text = re.sub(r"'''", '**', text)
    text = re.sub(r"''", '*', text)
    # Remove HTML tags & comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    # Clean up double/triple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class LoreExtractor:
    def __init__(self, output_dir: Optional[Path] = None, lang: str = LORE_LANG):
        self.output_dir = output_dir or WEB_LORE_DIR
        self.lang = lang
        self.champions_dir = self.output_dir / "champions"
        self.regions_dir = self.output_dir / "regions"
        self.stories_dir = self.output_dir / "stories"

        for d in [self.champions_dir, self.regions_dir, self.stories_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._ddragon_version: Optional[str] = None

    def get_latest_ddragon_version(self) -> str:
        if not self._ddragon_version:
            try:
                resp = requests.get(DDRAGON_VERSIONS_URL, timeout=10)
                resp.raise_for_status()
                self._ddragon_version = resp.json()[0]
                logger.info(f"Latest Data Dragon version: {self._ddragon_version}")
            except Exception as e:
                logger.warning(f"Failed to fetch DDragon version ({e}). Falling back to 14.22.1")
                self._ddragon_version = "14.22.1"
        return self._ddragon_version

    # =========================================================================
    # 1. Clean up legacy gameplay files
    # =========================================================================
    def purge_gameplay_files(self):
        """Removes obsolete champion files that contain gameplay kit descriptions."""
        count = 0
        for f in self.champions_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                # If it has spells or tips, remove it
                if "spells" in data or "Habilidades e Conceitos" in data.get("text", ""):
                    md_file = f.with_suffix(".md")
                    f.unlink(missing_ok=True)
                    md_file.unlink(missing_ok=True)
                    count += 1
            except (json.JSONDecodeError, OSError) as e:
                logger.debug(f"Error checking legacy file {f}: {e}")
        if count > 0:
            logger.info(f"Purged {count} legacy files containing gameplay kit data.")

    # =========================================================================
    # 2. Champion Biographies (Data Dragon - Pure Narrative)
    # =========================================================================
    def extract_champions_ddragon(self, limit: Optional[int] = None, specific_champion: Optional[str] = None) -> List[Dict]:
        """
        Extracts official narrative champion biographies from Riot Data Dragon.
        Does NOT extract spells, passives, abilities, or gameplay tips.
        """
        version = self.get_latest_ddragon_version()
        url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/{self.lang}/champion.json"
        logger.info(f"Fetching champion index from: {url}")

        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            champions_data = resp.json().get("data", {})
        except Exception as e:
            logger.error(f"Error fetching champions list: {e}")
            return []

        champ_keys = list(champions_data.keys())
        if specific_champion:
            champ_keys = [k for k in champ_keys if k.lower() == specific_champion.lower()]
            if not champ_keys:
                logger.error(f"Champion '{specific_champion}' not found in Data Dragon.")
                return []
        elif limit:
            champ_keys = champ_keys[:limit]

        logger.info(f"Extracting pure narrative lore for {len(champ_keys)} champion(s) in '{self.lang}'...")
        extracted = []

        for key in champ_keys:
            detail_url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/{self.lang}/champion/{key}.json"
            try:
                c_resp = requests.get(detail_url, timeout=10)
                c_resp.raise_for_status()
                c_data = c_resp.json().get("data", {}).get(key, {})
            except Exception as e:
                logger.warning(f"Could not fetch details for {key}: {e}")
                continue

            name = c_data.get("name", key)
            title = c_data.get("title", "")
            lore = c_data.get("lore", "")
            tags = c_data.get("tags", [])

            if not lore.strip():
                continue

            # Format pure markdown document - ZERO gameplay kit
            md_lines = [
                f"# {name} - {title}",
                f"**Tipo:** Biografia Canônica de Campeão | **Idioma:** {self.lang}",
                f"**Facção / Arquétipo:** {', '.join(tags)}",
                "",
                "## História e Biografia Oficial",
                lore,
            ]
            md_content = "\n".join(md_lines)

            record = {
                "id": f"champ_bio_{sanitize_filename(key).lower()}_{self.lang}",
                "source": "Riot Data Dragon (Official Narrative)",
                "category": "champion_bio",
                "name": name,
                "title": title,
                "language": self.lang,
                "tags": tags,
                "word_count": len(lore.split()),
                "text": md_content,
            }

            clean_filename = sanitize_filename(f"{key}_bio_{self.lang}")
            with open(self.champions_dir / f"{clean_filename}.json", "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            with open(self.champions_dir / f"{clean_filename}.md", "w", encoding="utf-8") as f:
                f.write(md_content)

            extracted.append(record)
            time.sleep(0.05)

        logger.info(f"Saved {len(extracted)} pure champion biographies from Data Dragon.")
        return extracted

    # =========================================================================
    # 3. Deep Champion Lore & Canon Relations (League Wiki)
    # =========================================================================
    def extract_champion_wiki_lore(self, champ_name: str) -> Optional[Dict]:
        """
        Fetches the complete canonical history, background, personality, and lore relations
        of a champion from the League of Legends Wiki using wikitext parsing.
        """
        params = {
            "action": "parse",
            "page": champ_name,
            "prop": "wikitext",
            "format": "json",
            "redirects": 1,
        }
        try:
            resp = requests.get(WIKI_API_URL, params=params, headers=WIKI_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Error requesting Wiki lore for {champ_name}: {e}")
            return None

        parse = data.get("parse")
        if not parse:
            return None

        actual_title = parse.get("title", champ_name)
        wikitext = parse.get("wikitext", {}).get("*", "")
        if not wikitext:
            return None

        # Split into level 2 sections
        sections = re.split(r'\n==\s*([^=]+)\s*==\n', wikitext)
        lore_parts = []
        wanted_headers = ["background", "personality", "relations", "appearance", "biography", "history"]

        for i in range(1, len(sections), 2):
            header = sections[i].strip()
            content = sections[i + 1] if i + 1 < len(sections) else ""
            if any(w in header.lower() for w in wanted_headers):
                cleaned = clean_wikitext(content)
                # Convert subheadings === to ###, ==== to ####
                cleaned = re.sub(r'===\s*([^=]+)\s*===', r'### \1', cleaned)
                cleaned = re.sub(r'====\s*([^=]+)\s*====', r'#### \1', cleaned)
                lore_parts.append(f"## {header}\n\n{cleaned}")

        if not lore_parts:
            return None

        full_lore = "\n\n".join(lore_parts)
        word_count = len(full_lore.split())
        if word_count < 100:
            return None

        md_lines = [
            f"# História Completa: {actual_title}",
            f"**Tipo:** Lore Aprofundada & Relações Canônicas | **Fonte:** League of Legends Universe / Wiki",
            "",
            full_lore,
        ]
        md_content = "\n".join(md_lines)

        record = {
            "id": f"champ_wiki_{sanitize_filename(actual_title).lower()}",
            "source": "League Wiki Canonical Lore",
            "category": "champion_wiki_lore",
            "name": actual_title,
            "title": f"Lore Canônica: {actual_title}",
            "word_count": word_count,
            "text": md_content,
        }

        clean_filename = sanitize_filename(f"{actual_title}_wiki_lore")
        with open(self.champions_dir / f"{clean_filename}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        with open(self.champions_dir / f"{clean_filename}.md", "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f" - Saved Wiki Lore: '{actual_title}' ({word_count} words)")
        return record

    def extract_champions_wiki(self, champions_list: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict]:
        """Extracts deep canonical lore for a list of major lore champions."""
        target_champions = champions_list or MAJOR_LORE_CHAMPIONS
        if limit:
            target_champions = target_champions[:limit]

        logger.info(f"Extracting deep Wiki canonical lore for {len(target_champions)} champion(s)...")
        extracted = []

        for champ in target_champions:
            rec = self.extract_champion_wiki_lore(champ)
            if rec:
                extracted.append(rec)
            time.sleep(0.3)

        logger.info(f"Total deep champion lore documents extracted: {len(extracted)}")
        return extracted

    # =========================================================================
    # 4. Region Lore Extraction (League Wiki)
    # =========================================================================
    def extract_regions(self, regions: Optional[List[str]] = None) -> List[Dict]:
        """Extracts world lore for regions of Runeterra."""
        target_regions = regions or REGIONS
        logger.info(f"Extracting world lore for {len(target_regions)} region(s)...")
        extracted = []

        for region in target_regions:
            page_title = region
            params = {
                "action": "parse",
                "page": page_title,
                "format": "json",
                "prop": "text",
                "redirects": 1,
            }

            try:
                resp = requests.get(WIKI_API_URL, params=params, headers=WIKI_HEADERS, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"Error fetching region {region}: {e}")
                continue

            parse = data.get("parse")
            if not parse:
                continue

            html = parse.get("text", {}).get("*", "")
            soup = BeautifulSoup(html, "html.parser")

            for tag in soup(["script", "style", "nav", "aside", "table", "noscript"]):
                tag.decompose()

            raw_text = soup.get_text()
            paragraphs = [p.strip() for p in raw_text.splitlines() if len(p.strip()) > 40]
            clean_body = "\n\n".join(paragraphs)

            if len(clean_body.split()) < 100:
                continue

            md_lines = [
                f"# Região: {region}",
                f"**Tipo:** Lore de Mundo & Geografia de Runeterra | **Fonte:** League of Legends Universe Wiki",
                "",
                clean_body,
            ]
            md_content = "\n".join(md_lines)

            record = {
                "id": f"region_{sanitize_filename(region).lower()}",
                "source": "League Wiki World Lore",
                "category": "region_lore",
                "name": region,
                "title": f"Região: {region}",
                "word_count": len(clean_body.split()),
                "text": md_content,
            }

            clean_filename = sanitize_filename(region)
            with open(self.regions_dir / f"{clean_filename}.json", "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            with open(self.regions_dir / f"{clean_filename}.md", "w", encoding="utf-8") as f:
                f.write(md_content)

            extracted.append(record)
            time.sleep(0.3)

        logger.info(f"Total region lore documents saved: {len(extracted)}")
        return extracted

    # =========================================================================
    # 5. Universe Canonical Short Stories Extraction
    # =========================================================================
    def extract_short_stories(self, limit: int = 60) -> List[Dict]:
        """
        Extracts official canonical short stories from 'Category:Short Story' on League Wiki.
        """
        logger.info(f"Extracting official Universe short stories (limit {limit})...")
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:Short Story",
            "cmlimit": min(limit, 500),
            "format": "json",
        }

        try:
            resp = requests.get(WIKI_API_URL, params=params, headers=WIKI_HEADERS, timeout=15)
            resp.raise_for_status()
            members = resp.json().get("query", {}).get("categorymembers", [])
        except Exception as e:
            logger.error(f"Error fetching short story category: {e}")
            return []

        story_titles = [m["title"] for m in members if not m["title"].startswith(("Category:", "File:", "Template:"))]
        story_titles = story_titles[:limit]
        logger.info(f"Found {len(story_titles)} story titles to extract...")

        extracted = []
        for title in story_titles:
            p_params = {
                "action": "parse",
                "page": title,
                "prop": "wikitext",
                "format": "json",
                "redirects": 1,
            }
            try:
                p_resp = requests.get(WIKI_API_URL, params=p_params, headers=WIKI_HEADERS, timeout=15)
                p_resp.raise_for_status()
                data = p_resp.json()
            except Exception as e:
                logger.warning(f"Error fetching story '{title}': {e}")
                continue

            parse = data.get("parse")
            if not parse:
                continue

            wikitext = parse.get("wikitext", {}).get("*", "")
            if not wikitext:
                continue

            clean_story = clean_wikitext(wikitext)
            word_count = len(clean_story.split())
            if word_count < 150:
                continue

            md_lines = [
                f"# Conto: {title}",
                f"**Tipo:** Conto Canônico Oficial do Universo | **Fonte:** League of Legends Universe",
                "",
                clean_story,
            ]
            md_content = "\n".join(md_lines)

            record = {
                "id": f"story_{sanitize_filename(title).lower()}",
                "source": "League of Legends Universe Short Story",
                "category": "short_story",
                "title": title,
                "word_count": word_count,
                "text": md_content,
            }

            clean_filename = sanitize_filename(f"Conto_{title}")[:80]
            with open(self.stories_dir / f"{clean_filename}.json", "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            with open(self.stories_dir / f"{clean_filename}.md", "w", encoding="utf-8") as f:
                f.write(md_content)

            extracted.append(record)
            logger.info(f" - Story saved: '{title}' ({word_count} words)")
            time.sleep(0.3)

        logger.info(f"Total canonical short stories extracted: {len(extracted)}")
        return extracted


def main():
    parser = argparse.ArgumentParser(description="League of Legends Lore Extractor (Pure Narrative Lore)")
    parser.add_argument("--all", action="store_true", help="Extract all champions, wiki lore, regions, and stories")
    parser.add_argument("--purge", action="store_true", help="Purge legacy files with gameplay kit data")
    parser.add_argument("--ddragon", action="store_true", help="Extract pure champion biographies from Data Dragon")
    parser.add_argument("--wiki-champions", action="store_true", help="Extract deep wiki canonical lore for major champions")
    parser.add_argument("--regions", action="store_true", help="Extract region lore from League Wiki")
    parser.add_argument("--stories", action="store_true", help="Extract Universe canonical short stories")
    parser.add_argument("--limit-stories", type=int, default=50, help="Limit short stories (default 50)")
    parser.add_argument("--limit-wiki", type=int, default=None, help="Limit major wiki champions")
    parser.add_argument("--limit-ddragon", type=int, default=None, help="Limit ddragon champions")
    parser.add_argument("--lang", type=str, default=LORE_LANG, help="Language code (pt_BR, en_US)")

    args = parser.parse_args()
    extractor = LoreExtractor(lang=args.lang)

    # Always purge legacy files containing gameplay kit
    extractor.purge_gameplay_files()

    if args.all or args.ddragon:
        extractor.extract_champions_ddragon(limit=args.limit_ddragon)

    if args.all or args.wiki_champions:
        extractor.extract_champions_wiki(limit=args.limit_wiki)

    if args.all or args.regions:
        extractor.extract_regions()

    if args.all or args.stories:
        extractor.extract_short_stories(limit=args.limit_stories)


if __name__ == "__main__":
    main()
