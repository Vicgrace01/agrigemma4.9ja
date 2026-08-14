import csv
import re
import subprocess
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT / "naerls_sources"
OUTPUT = PROJECT / "naerls_verified.csv"

SOURCES = [
    {
        "crop": "cassava",
        "title": "NAERLS Cassava Production, Processing and Utilization in Nigeria",
        "url": "https://naerls.gov.ng/wp-content/uploads/2022/11/CASSAVA-Production-Processing-and-utilization.pdf",
    },
    {
        "crop": "maize",
        "title": "NAERLS Maize Production, Marketing, Processing and Utilization in Nigeria",
        "url": "https://naerls.gov.ng/wp-content/uploads/2022/11/MAIZE-Production-Marketing-Processing-Utilization-In-Nigeria.pdf",
    },
    {
        "crop": "rice",
        "title": "NAERLS Rice Production, Processing, Utilization and Marketing in Nigeria",
        "url": "https://naerls.gov.ng/wp-content/uploads/2022/11/Rice-Production-Processing-Utilization-and-Marketing-in-Nigeria.pdf",
    },
    {
        "crop": "tomato",
        "title": "NAERLS Heat-Tolerant Tomato Production Under Irrigation",
        "url": "https://naerls.gov.ng/wp-content/uploads/2022/11/Heat-Tolerant-Tomato-Production-Under-Irrigation.pdf",
    },
    {
        "crop": "pepper",
        "title": "NAERLS Irrigated Pepper Production",
        "url": "https://naerls.gov.ng/wp-content/uploads/2022/11/Irrigated-Pepper-Production.pdf",
    },
    {
        "crop": "onion",
        "title": "NAERLS Irrigated Onion Production and Management",
        "url": "https://naerls.gov.ng/wp-content/uploads/2022/11/Irrigated-Onion-Production-And-Management.pdf",
    },
]

STOPWORDS = {
    "about", "after", "also", "and", "are", "been", "being", "between",
    "but", "can", "for", "from", "has", "have", "into", "its", "more",
    "not", "of", "on", "or", "that", "the", "their", "this", "those",
    "through", "to", "under", "use", "used", "using", "was", "with",
}

def clean_text(text):
    text = text.replace("\x0c", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def meaningful_paragraphs(text):
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = clean_text(paragraph)
        letters = sum(char.isalpha() for char in paragraph)

        if letters < 80:
            continue

        if "table of content" in paragraph.lower():
            continue

        yield paragraph

def chunks(paragraphs, min_chars=450, max_chars=1100):
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

def keywords(crop, content):
    words = re.findall(r"[a-z]{3,}", content.lower())
    unique = []
    seen = set()

    for word in [crop, *words]:
        if word not in STOPWORDS and word not in seen:
            seen.add(word)
            unique.append(word)

        if len(unique) == 45:
            break

    return " ".join(unique)

def download(source, destination):
    if destination.exists() and destination.stat().st_size > 100_000:
        print(f"Using existing: {destination.name}")
        return

    print(f"Downloading: {destination.name}")
    request = urllib.request.Request(
        source["url"],
        headers={"User-Agent": "AgriGemma4.9ja-NAERLS-Corpus-Builder/1.0"},
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        destination.write_bytes(response.read())

def extract_pages(pdf_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "page"

    subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), str(prefix) + ".txt"],
        check=True,
    )

    raw_text = (output_dir / "page.txt").read_text(encoding="utf-8", errors="replace")
    return raw_text.split("\f")

def main():
    SOURCE_DIR.mkdir(exist_ok=True)
    rows = []
    record_number = 1

    for source in SOURCES:
        filename = re.sub(r"[^a-z0-9]+", "_", source["crop"].lower()).strip("_") + ".pdf"
        pdf_path = SOURCE_DIR / filename
        download(source, pdf_path)

        pages = extract_pages(pdf_path, SOURCE_DIR / f"{source['crop']}_text")

        for page_number, page_text in enumerate(pages, start=1):
            cleaned = clean_text(page_text)

            for chunk in chunks(meaningful_paragraphs(cleaned)):
                rows.append(
                    {
                        "id": f"naerls_{record_number:04d}",
                        "crop": source["crop"],
                        "topic": "official_extension_bulletin",
                        "keywords": keywords(source["crop"], chunk),
                        "content": chunk,
                        "source_title": source["title"],
                        "source_url": source["url"],
                        "source_page": page_number,
                        "verified": "official_naerls_bulletin",
                    }
                )
                record_number += 1

    if not rows:
        raise SystemExit("No records created. Check network access and pdftotext.")

    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCreated {OUTPUT.name} with {len(rows)} verified NAERLS evidence records.")
    print("Original naerls_database.csv was not changed.")

if __name__ == "__main__":
    main()
