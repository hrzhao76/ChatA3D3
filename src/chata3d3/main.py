import panel as pn
from uuid import uuid4
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import CallbackManager
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from chata3d3.config import (
    MODEL_DIR,
    MODEL_FILENAME,
    RAG_STORES_PATH,
    EMBEDDING_MODEL_ID,
    STATIC_DIR,
)

pn.extension()

use_rag_toggle = pn.widgets.Checkbox(name="Use RAG", value=True)

qdrant_client = QdrantClient(path=RAG_STORES_PATH)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="a3d3-knowledge",
    embedding=embedding_model,
)

rag_template = (
    "You are a helpful assistant. Answer the following question in complete "
    "sentences based only on the given context. Do not copy any multiple "
    "choice format.\n\n"
    "{context}\n\n"
    "Question: {question}"
)

no_rag_template = """You are a helpful assistant. Please answer the question.

Question: {question}"""


def get_chain(callback_handlers, use_rag: bool):
    callback_manager = CallbackManager(callback_handlers)

    llm = LlamaCpp(
        model_path=str(MODEL_DIR / MODEL_FILENAME),
        callback_manager=callback_manager,
        temperature=0.8,
        n_ctx=2048,
        max_tokens=512,
        n_gpu_layers=-1,
        verbose=False,
    )

    if use_rag:
        retriever = vector_store.as_retriever(
            callbacks=callback_handlers,
            search_type="mmr",
            search_kwargs={"k": 2},
        )

        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        def show_docs(docs):
            for h in callback_handlers:
                h.on_retriever_end(docs, run_id=uuid4())
            return docs

        return (
            {
                "context": retriever | show_docs | format_docs,
                "question": RunnablePassthrough(),
            }
            | PromptTemplate.from_template(rag_template)
            | llm
        )
    else:
        return (
            {
                "context": lambda _: "",
                "question": RunnablePassthrough(),
            }
            | PromptTemplate.from_template(no_rag_template)
            | llm
        )


async def callback(contents, user, instance):
    handler = pn.chat.langchain.PanelCallbackHandler(
        instance, user="A3D3 ChatBot", avatar=str(STATIC_DIR / "A3D3-logo.png")
    )
    handler.on_llm_end = lambda *_: None

    chain = get_chain([handler], use_rag_toggle.value)
    await chain.ainvoke(contents)


chat_interface = pn.chat.ChatInterface(callback=callback)
dashboard = pn.Column(use_rag_toggle, chat_interface)

pn.serve({"/": dashboard}, port=5006, websocket_origin="*", show=False)
