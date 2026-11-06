import urllib.request
import urllib.parse
import json
import re

def clean_wikitext(text: str) -> str:
    # {{cbi|Kalista}} -> Kalista
    text = re.sub(r'\{\{(?:cbi|ci|ai|bi|ri|sbi)\|([^}|]+)(?:\|[^}]+)?\}\}', r'\1', text, flags=re.IGNORECASE)
    # Generic templates {{...}} -> empty
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    # Wikilinks [[Target|Display]] -> Display, [[Target]] -> Target
    text = re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', text)
    # Bold / Italic
    text = re.sub(r"'''", '**', text)
    text = re.sub(r"''", '*', text)
    # HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def fetch_champion_wiki_lore(champ_name: str):
    params = {'action': 'parse', 'page': champ_name, 'prop': 'wikitext', 'format': 'json'}
    url = 'https://leagueoflegends.fandom.com/api.php?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': 'RuneterraLoreRAG/1.0'})
    res = urllib.request.urlopen(req, timeout=12)
    data = json.loads(res.read().decode())
    wikitext = data.get('parse', {}).get('wikitext', {}).get('*', '')

    sections = re.split(r'\n==\s*([^=]+)\s*==\n', wikitext)
    lore_parts = []
    wanted = ['background', 'personality', 'relations', 'appearance', 'biography', 'history']

    for i in range(1, len(sections), 2):
        header = sections[i].strip()
        content = sections[i+1] if i+1 < len(sections) else ''
        if any(w in header.lower() for w in wanted):
            clean_content = clean_wikitext(content)
            clean_content = re.sub(r'===\s*([^=]+)\s*===', r'### \1', clean_content)
            clean_content = re.sub(r'====\s*([^=]+)\s*====', r'#### \1', clean_content)
            lore_parts.append(f"## {header}\n\n{clean_content}")

    full_lore = "\n\n".join(lore_parts)
    return full_lore

if __name__ == "__main__":
    for c in ["Viego", "Garen", "Mordekaiser", "Aatrox", "Swain"]:
        lore = fetch_champion_wiki_lore(c)
        words = len(lore.split())
        print(f"{c}: {words} words extracted successfully.")
