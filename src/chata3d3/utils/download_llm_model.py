import requests
from pathlib import Path
from tqdm import tqdm

from chata3d3.config import setup_logging, MODEL_FILENAME, MODEL_DIR, MODEL_URL

import logging

setup_logging()
logger = logging.getLogger(__name__)


def download_llm_model(model_dir: str):
    """Download the model file if not present in model_dir."""
    model_path = Path(model_dir) / MODEL_FILENAME

    if model_path.exists():
        logging.info(f"✅ Model already exists at: {model_path}")
        return model_path

    logging.info(f"⬇️ Model not found at {model_path}, downloading from HuggingFace...")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(model_path, "wb") as f,
            tqdm(
                desc=f"Downloading {MODEL_FILENAME}",
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    logging.info(f"✅ Download completed: {model_path}")
    return model_path


if __name__ == "__main__":
    download_llm_model(model_dir=MODEL_DIR)
