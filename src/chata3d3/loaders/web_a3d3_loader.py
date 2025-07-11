import csv
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from chata3d3.config import setup_logging, DATA_DIR, HTML_CSV_PATH

setup_logging()
logger = logging.getLogger(__name__)

headers = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    )
}


def is_text_url(url):
    non_text_ext = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".svg",
        ".pdf",
        ".zip",
        ".js",
        ".css",
        ".mp4",
        ".webm",
        ".ttf",
        ".woff",
        ".ico",
        ".webp",
    )
    return not url.lower().endswith(non_text_ext)


def extract_urls_from_sitemap(url):
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "xml")
        return [loc.text for loc in soup.find_all("loc")]
    except Exception as e:
        logger.warning(f"❌ Failed to extract from {url}: {e}")
        return []


def extract_a3d3_web_sitemaps():
    sitemap_index = "https://a3d3.ai/sitemap_index.xml"
    logger.info(f"🌐 Fetching sitemap index: {sitemap_index}")
    urls = extract_urls_from_sitemap(sitemap_index)

    all_urls = []
    # TODO data clean
    for sitemap_url in urls[0:2]:
        logger.info(f"📂 Parsing sitemap: {sitemap_url}")
        page_urls = extract_urls_from_sitemap(sitemap_url)
        all_urls.extend(page_urls)

    filtered = [url for url in all_urls if is_text_url(url)]
    logger.info(
        f"✅ {len(filtered)} text-like pages after filtering from {len(all_urls)} total."
    )
    return filtered


def create_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(headers)
    return session


def download_single_page(idx_url_pair, data_path, session):
    idx, url = idx_url_pair
    try:
        res = session.get(url, timeout=15)
        res.raise_for_status()
        filename = data_path / f"{idx:04}.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(res.text)
        time.sleep(0.5)  # ⏳ 限速
        return {"url": url, "filename": str(filename)}
    except Exception as e:
        logger.warning(f"❌ Failed to download {url}: {e}")
        return None


def download_web_from_sitemaps(
    csv_path: Path, data_path: Path, urls=None, max_workers=8
):
    if urls is None:
        logger.error("❌ No URLs provided to download.")
        return

    data_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"⏬ Downloading {len(urls)} pages to: {data_path} with {max_workers} threads..."
    )

    session = create_session()
    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_single_page, (idx, url), data_path, session)
            for idx, url in enumerate(urls)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:
                records.append(result)

    logger.info(f"📄 Writing metadata to: {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "filename"])
        writer.writeheader()
        writer.writerows(records)

    logger.info("✅ Done downloading and saving metadata.")


if __name__ == "__main__":
    html_data_dir = DATA_DIR / "htmls"
    html_csv_path = HTML_CSV_PATH

    urls = extract_a3d3_web_sitemaps()
    download_web_from_sitemaps(
        csv_path=html_csv_path, data_path=html_data_dir, urls=urls
    )
