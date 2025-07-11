import re
import logging

from langchain_community.document_loaders import BSHTMLLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

from chata3d3.config import (
    setup_logging,
    PDF_DIR,
    HTML_DIR,
    EMBEDDING_MODEL_DIR,
    EMBEDDING_MODEL_ID,
    RAG_STORES_PATH,
)

setup_logging()
logger = logging.getLogger(__name__)


def clean_docs(docs):
    cleaned = []
    for doc in docs:
        text = doc.page_content
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"©.*?Ltd", "", text)
        text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub("[\ue000-\uf8ff]", "", text)
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"\t+", "\t", text)
        text = re.sub(r" +", " ", text)
        doc.page_content = text.strip()
        cleaned.append(doc)
    return cleaned


def build_rag_vector_store(store_path, embedding_model_path):
    logger.info("📄 Loading PDF documents...")
    pdf_docs = []
    for path in sorted(PDF_DIR.glob("*.pdf")):
        loader = PyMuPDFLoader(str(path))
        pdf_docs.extend(loader.load())

    logger.info("📄 Loading HTML documents...")
    html_docs = []
    for path in sorted(HTML_DIR.glob("*.html")):
        loader = BSHTMLLoader(str(path))
        try:
            html_docs.extend(loader.load())
        except Exception as e:
            logger.warning(f"❌ Failed to load {path}: {e}")

    logger.info("🧼 Cleaning documents...")
    docs = clean_docs(pdf_docs + html_docs)

    logger.info("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)

    logger.info(f"🧠 Loading embedding model from: {embedding_model_path}")
    embeddings_model = HuggingFaceEmbeddings(model_name=str(embedding_model_path))
    vector_size = len(embeddings_model.embed_query("test"))

    logger.info(f"🧱 Creating Qdrant collection at {store_path}...")
    client = QdrantClient(path=str(store_path))

    collection_name = "a3d3-knowledge"
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings_model,
    )

    logger.info("📦 Adding documents to Qdrant...")
    document_ids = vector_store.add_documents(chunks)

    logger.info(
        f"✅ Stored {len(document_ids)} chunks in collection '{collection_name}'."
    )


if __name__ == "__main__":
    build_rag_vector_store(
        store_path=RAG_STORES_PATH,
        embedding_model_path=EMBEDDING_MODEL_DIR / EMBEDDING_MODEL_ID,
    )
