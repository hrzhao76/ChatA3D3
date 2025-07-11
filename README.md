# ChatA3D3

A chatbot powered by a Large Language Model for the A3D3 collaboration.
This is a **Minimum Viable Product (MVP)** version with **RAG (Retrieval-Augmented Generation)** support.

## 🔍 Data Sources

The chatbot is powered by documents collected from the following sources:

- 🌐 [A3D3 official website](https://a3d3.ai)
- 📄 NSF award pages and related papers mentioning A3D3-funded research

### 📊 Data Summary

The collected and processed data are summarized in the following CSV files:

- `data/a3d3_webs.csv`: Metadata of pages downloaded from the A3D3 website
- `data/nsf_award_papers_filtered.csv`: Cleaned list of papers associated with NSF awards relevant to A3D3
- `data/failed_pdfs.csv`: Records of papers that could not be accessed or downloaded

> ⚠️ Note: Some NSF-linked papers are not publicly accessible and are tracked in the `failed_pdfs.csv` file.

## 🧱 Tech Stack

- **LLM Backend**: [`llama.cpp[gpu]`](https://github.com/ggerganov/llama.cpp) serving `OLMo-7B-Instruct-Q4_K_M.gguf` on GPU
- **Vector Store**: [Qdrant](https://qdrant.tech/) for semantic document retrieval
- **Orchestration**: [LangChain](https://www.langchain.com/) for chain composition and prompt handling
- **Frontend**: [Panel](https://panel.holoviz.org/) for interactive chatbot UI

## ✅ Current Features

- 🧠 Supports switching between pure LLM and RAG mode (via “Use RAG” checkbox)
- 🗂 Uses HuggingFace embeddings + Qdrant vector store for document retrieval
- 🖥 Interactive web UI with [Panel ChatInterface](https://panel.holoviz.org/user_guide/ChatInterface.html)
- Pricavy and Local First

## 🖼️ Screenshot

Here are screenshots of the Panel-based chatbot UI:
### Chat with RAG
![ChatA3D3 UI](./assets/chatA3D3-rag.png)

### Chat without RAG
![ChatA3D3 UI - RAG disabled](./assets/chatA3D3-norag.png)


## 🚧 TODOs

- [ ] Deploy on the UCSD National Research Platform (NRP)
- [ ] Enhance HTML cleaning and document parsing from the A3D3 website
- [ ] Try different models. Finetune maybe.
- [ ] Add support for OpenWebUI / Ollama-compatible backends

## 📁 Project Structure (WIP)

```
src/chata3d3
├── config.py             # Constants: model paths, store locations
├── langchain             # (Optional) LangChain extensions and integrations
├── loaders               # Web + NSF data fetchers
├── main.py               # Panel-based chat interface entry point
├── rag                   # RAG chain, vector store, retriever setup
├── static                # Static assets (optional)
└── utils                 # Utility scripts and model downloaders

```

## 📦 Setup Notes

```sh
# 1. Clone the repository
git clone https://github.com/hrzhao76/ChatA3D3.git
cd ChatA3D3

# 2. Install dependencies
pip install -e .

# 3. Install llama-cpp-python with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python

# 4. Download the LLM and embedding model
python -m chata3d3.utils.download_llm_model
python -m chata3d3.utils.download_embedding_model

# 5. Fetch documents
python -m chata3d3.loaders.paper_nsf_loader
python -m chata3d3.loaders.web_a3d3_loader

# 6. Build the vector store
python -m chata3d3.rag.build_rag_vector_store

# 7. Launch the chatbot
python -m chata3d3.main
```
