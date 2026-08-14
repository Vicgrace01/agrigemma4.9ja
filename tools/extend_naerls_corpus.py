import csv
import re
import subprocess
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT / "naerls_sources"
CORPUS = PROJECT / "naerls_verified.csv"
BULLETINS_URL = "https://naerls.gov.ng/bulletins/"
DISEASE_URL = "https://kms.naerls.gov.ng/diseasemanagment/listdiseases"

TARGET_TITLES = (
    "bull and ram fattening",
    "herd health management",
    "control of worms in cattle sheep and goats",
    "duck production",
    "economics of aquaculture",
    "feed formulation and feeding practices in nigeria fish culture",
    "fish culture in ponds",
    "fish pond fertilization",
    "fish pond site selection",
    "hatchery management practices in poultry",
    "improving the performance of local chickens",
    "integrated aquaculture technologies",
    "management of drugs and veterinary equipment",
    "production and utilization of ogbono",
    "production of lablab",
    "production of gum arabic",
    "rubber production",
    "snail production",
    "production of guinea fowl",
)

STOPWORDS = {
    "about", "after", "also", "and", "are", "been", "being", "between",
    "but", "can", "for", "from", "has", "have", "into", "its", "more",
    "not", "of", "on", "or", "that", "the", "their", "this", "those",
    "through", "to", "under", "use", "used", "using", "was", "with",
}

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.href = None
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.text = []

    def handle_data(self, data):
        if self.href:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            label = " ".join("".join(self.text).split())
            self.links.append((label, self.href))
            self.href = None
            self.text = []

class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.in_cell = False
        self.cell = []
        self.row = []

    def handle_starttag(self, tag, attrs):
        if tag in {"td", "th"}:
            self.in_cell = True
            self.cell = []

    def handle_data(self, data):
        if self.in_cell:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.in_cell:
            self.row.append(" ".join("".join(self.cell).split()))
            self.in_cell = False
        elif tag == "tr":
            cleaned = [cell for cell in self.row if cell]
            if cleaned:
                self.rows.append(cleaned)
            self.row = []

def fetch(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AgriGemma4.9ja-NAERLS-Corpus-Builder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()

def clean_text(text):
    text = text.replace("\x0c", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def meaningful_paragraphs(text):
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = clean_text(paragraph)
        if sum(char.isalpha() for char in paragraph) >= 80:
            yield paragraph

def chunks(paragraphs, min_chars=400, max_chars=1000):
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current} {paragraph}".strip()
        if current and len(candidate) > max_chars:
            if len(current) >= min_chars:
                yield current
                current = paragraph
            else:
                current = candidate
        else:
            current = candidate

    if len(current) >= min_chars:
        yield current

def make_keywords(topic, content):
    words = re.findall(r"[a-z]{3,}", f"{topic} {content}".lower())
    result, seen = [], set()

    for word in words:
        if word not in STOPWORDS and word not in seen:
            seen.add(word)
            result.append(word)
        if len(result) >= 45:
            break

    return " ".join(result)

def safe_name(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:70]

def load_existing():
    with CORPUS.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))

def main():
    SOURCE_DIR.mkdir(exist_ok=True)
    rows = load_existing()
    existing_urls = {row.get("source_url", "") for row in rows}
    next_id = len(rows) + 1

    bulletin_page = fetch(BULLETINS_URL).decode("utf-8", errors="replace")
    parser = LinkParser()
    parser.feed(bulletin_page)

    selected = []
    for title, url in parser.links:
        normalized = title.lower()
        if any(target in normalized for target in TARGET_TITLES) and url.lower().endswith(".pdf"):
            if url not in existing_urls:
                selected.append((title, url))

    print(f"Found {len(selected)} additional official NAERLS bulletins.")

    for title, url in selected:
        pdf_path = SOURCE_DIR / f"{safe_name(title)}.pdf"

        try:
            if not pdf_path.exists() or pdf_path.stat().st_size < 100_000:
                print(f"Downloading: {title}")
                pdf_path.write_bytes(fetch(url))

            text_path = SOURCE_DIR / f"{safe_name(title)}.txt"
            subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), str(text_path)],
                check=True,
                capture_output=True,
            )

            pages = text_path.read_text(encoding="utf-8", errors="replace").split("\f")

            for page_number, page in enumerate(pages, start=1):
                for content in chunks(meaningful_paragraphs(page)):
                    rows.append(
                        {
                            "id": f"naerls_{next_id:04d}",
                            "crop": "livestock_fisheries_or_specialty_crop",
                            "topic": title,
                            "keywords": make_keywords(title, content),
                            "content": clean_text(content),
                            "source_title": title,
                            "source_url": url,
                            "source_page": page_number,
                            "verified": "official_naerls_bulletin",
                        }
                    )
                    next_id += 1

            existing_urls.add(url)

        except Exception as error:
            print(f"Skipped {title}: {error}")

    try:
        disease_page = fetch(DISEASE_URL).decode("utf-8", errors="replace")
        disease_parser = TableParser()
        disease_parser.feed(disease_page)

        disease_rows = 0
        for cells in disease_parser.rows:
            joined = " | ".join(cells)

            if len(cells) < 2 or len(joined) < 30:
                continue

            if not re.search(
                r"disease|rot|blight|mosaic|spot|mildew|wilt|worm|army|virus|fung",
                joined,
                flags=re.I,
            ):
                continue

            content = (
                "NAERLS Farmer Knowledge Base disease-management entry: "
                + joined
            )

            rows.append(
                {
                    "id": f"naerls_{next_id:04d}",
                    "crop": "crop_disease_management",
                    "topic": "NAERLS crop disease catalogue",
                    "keywords": make_keywords("crop disease pest control", content),
                    "content": content,
                    "source_title": "NAERLS Farmer Knowledge Base: Crop Disease Management",
                    "source_url": DISEASE_URL,
                    "source_page": "",
                    "verified": "official_naerls_kms_catalogue",
                }
            )
            next_id += 1
            disease_rows += 1

        print(f"Added {disease_rows} NAERLS disease-catalogue entries.")

    except Exception as error:
        print(f"Skipped disease catalogue: {error}")

    fieldnames = [
        "id", "crop", "topic", "keywords", "content",
        "source_title", "source_url", "source_page", "verified",
    ]

    with CORPUS.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCorpus now contains {len(rows)} verified NAERLS records.")

if __name__ == "__main__":
    main()
