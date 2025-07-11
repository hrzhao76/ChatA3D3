import logging
from pathlib import Path

from huggingface_hub import snapshot_download
from chata3d3.config import setup_logging, EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_ID

setup_logging()
logger = logging.getLogger(__name__)


def download_embedding_model(model_dir: str, model_id: str):
    """Download a HuggingFace model to a specific directory."""
    model_path = Path(model_dir) / model_id
    if (model_path / "config.json").exists():
        logger.info(f"✅ Embedding model already exists at: {model_path}")
        return model_path

    logger.info(f"⬇️ Downloading embedding model from {model_id} to {model_path}...")
    model_path.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_id,
        local_dir=model_path,
        local_dir_use_symlinks=False,
        token=None,
        ignore_patterns=["*.msgpack", "*.arrow"],
    )

    logger.info(f"✅ Download completed: {model_path}")
    return model_path


if __name__ == "__main__":
    download_embedding_model(EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_ID)
