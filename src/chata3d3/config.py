from pathlib import Path
import logging

# === Global constants ===
ROOT_DIR = Path(__file__).resolve().parents[2]  # return to the root dir
DATA_DIR = ROOT_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
HTML_DIR = DATA_DIR / "htmls"
PAPER_CSV_PATH = DATA_DIR / "nsf_award_papers_filtered.csv"
HTML_CSV_PATH = DATA_DIR / "a3d3_webs.csv"
FAILED_LOG = DATA_DIR / "failed_pdfs.csv"

# === MODEL constants ===
MODEL_FILENAME = "OLMo-7B-Instruct-Q4_K_M.gguf"
MODEL_DIR = ROOT_DIR / "model"
MODEL_URL = (
    "https://huggingface.co/ssec-uw/OLMo-7B-Instruct-GGUF/resolve/main/"
    "OLMo-7B-Instruct-Q4_K_M.gguf?download=true"
)
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L12-v2"
EMBEDDING_MODEL_DIR = MODEL_DIR

# === RAG constants ===
RAG_STORES_PATH = ROOT_DIR / "rag"

STATIC_DIR = ROOT_DIR / "src" / "chata3d3" / "static"


# === Logging config ===
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
