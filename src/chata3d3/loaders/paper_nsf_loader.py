import sys
import re
import csv
from pathlib import Path
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup

from chata3d3.config import setup_logging, DATA_DIR, PAPER_CSV_PATH
import logging

setup_logging()
logger = logging.getLogger(__name__)


def fetch_nsf_award_html(award_id: str) -> Optional[str]:
    url = f"https://www.nsf.gov/awardsearch/showAward?AWD_ID={award_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    logger.info(f"Fetching NSF Award page: {url}")
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        return res.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch NSF award page: {e}")
        return None


def parse_nsf_award_html(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    divs_raw = soup.find_all("div", class_="margintop15")
    divs = []

    for div in divs_raw:
        if div.find("div", id="showC1"):
            continue
        if any(
            re.search(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", a["href"])
            for a in div.find_all("a", href=True)
        ):
            divs.append(div)

    logger.info(f"Found {len(divs)} paper entries with DOIs.")
    rows = []

    for i, div in enumerate(divs):
        spans = div.find_all("span")
        paper_lines = [
            span.get_text(strip=True) for span in spans if span.get_text(strip=True)
        ]
        doi, doi_link, nsf_citation_id = "", "", ""

        for a in div.find_all("a", href=True):
            href = a["href"]
            if not doi and (
                match := re.search(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", href)
            ):
                doi = match.group(1)
                doi_link = f"https://doi.org/{doi}"
            if not nsf_citation_id and (
                match := re.search(r"par\.nsf\.gov/biblio/(\d+)", href)
            ):
                nsf_citation_id = match.group(1)

        title_cleaned = paper_lines[1].strip('"').strip()

        logger.debug(f"Paper {i+1}: {title_cleaned}")
        logger.debug(f"DOI: {doi_link}, NSF Citation ID: {nsf_citation_id}")

        rows.append(
            {
                "index": i + 1,
                "authors": paper_lines[0],  # authors
                "title": title_cleaned,  # title
                "journal": paper_lines[2],  # journal
                "doi": doi,
                "doi_link": doi_link,
                "nsf_citation_id": nsf_citation_id,
            }
        )

    return rows


def extract_nsf_paper_metadata(
    award_id="2117997", output_csv="downloads/nsf_award_papers_filtered.csv"
) -> Path:
    html = fetch_nsf_award_html(award_id)
    if html is None:
        logger.warning("No HTML retrieved; skipping parse.")
        return Path(output_csv)

    papers = parse_nsf_award_html(html)
    if len(papers) == 0:
        sys.exit("No CSV parsed!")

    keys = list(papers[0].keys())
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(papers)

    logger.info(f"Saved {len(papers)} entries to {output_path}")

    return output_path


def sanitize_filename(text):
    return re.sub(r"[^a-zA-Z0-9_\\-]", "_", text)[:80]


def download_pdf(url, filename):
    try:
        r = requests.get(url, timeout=15, stream=True)
        content_type = r.headers.get("Content-Type", "").lower()
        if "pdf" in content_type:
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        return False
    except Exception as e:
        print(f"⚠️ Error downloading PDF from {url}: {e}")
        return False


def download_pdfs_from_metadata(
    csv_path="downloads/nsf_award_papers_filtered.csv",
    data_path="downloads/",
    email="test@gmail.com",
):
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return

    output_dir = Path(data_path) / "pdfs"
    fail_log = Path(data_path) / "failed_pdfs.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_entries = []

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        papers = list(reader)

    for paper in papers:
        index = paper["index"]
        text = paper["title"]
        doi = paper["doi"]
        nsf_citation_id = paper["nsf_citation_id"]

        logging.info(f"📄 Processing paper {index}")
        short_title = sanitize_filename(text[:60])
        filename = output_dir / f"paper_{index}_{short_title}.pdf"

        if Path(filename).exists():
            logger.info(f"✅ Already downloaded: {filename}")
            continue
        is_success = False
        # Try Citation Details PDF first
        if nsf_citation_id:
            citation_pdf_url = f"https://par.nsf.gov/servlets/purl/{nsf_citation_id}"
            logger.debug(f"📥 Trying Citation Details PDF: {citation_pdf_url}")
            if download_pdf(citation_pdf_url, filename):
                logger.info(f"✅ Saved PDF from Citation Details: {filename}")
                is_success = True
                continue
            else:
                logger.warning("❌ Citation Details PDF failed.")

        # Try Unpaywall as fallback
        if not is_success and doi:
            logger.debug(f"🔍 Querying Unpaywall for DOI: {doi}")
            unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
            try:
                response = requests.get(unpaywall_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    oa = data.get("best_oa_location")
                    if oa and oa.get("url_for_pdf"):
                        pdf_url = oa["url_for_pdf"]
                        logger.debug(f"📥 Trying Unpaywall PDF: {pdf_url}")
                        if download_pdf(pdf_url, filename):
                            logging.info(f"✅ Saved PDF from Unpaywall: {filename}")
                            is_success = True
                            continue
                        else:
                            logging.warning("❌ Unpaywall PDF failed.")
                    else:
                        logging.warning("❌ No PDF in Unpaywall.")
                else:
                    logging.warning(f"⚠️ Failed Unpaywall lookup for {doi}")
            except Exception as e:
                logging.warning(f"⚠️ Unpaywall error for {doi}: {e}")

        if not is_success:
            failed_entries.append(paper)

    # Save failed attempts

    if failed_entries:
        logging.info(f"Writing {len(failed_entries)} failed entries...")
        keys = list(failed_entries[0].keys())
        with open(fail_log, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(failed_entries)
        logging.warning(f"⚠️ Logged {len(failed_entries)} failed entries to {fail_log}")
    else:
        logging.info("🎉 All PDFs downloaded successfully!")


if __name__ == "__main__":
    extract_nsf_paper_metadata(output_csv=PAPER_CSV_PATH)
    download_pdfs_from_metadata(csv_path=PAPER_CSV_PATH, data_path=DATA_DIR)
