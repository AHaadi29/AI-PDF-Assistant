import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.config import CHROMA_DIR, EMBEDDING_MODEL_NAME

_embeddings = None
COLLECTION_NAME = "pdf_documents"


class SimpleDoc:
    """A minimal stand-in for a LangChain Document, so code that expects
    .page_content and .metadata works the same regardless of where the
    chunk came from."""

    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def get_embeddings():
    """Loads the AI embedding model once and reuses it."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


def get_vectorstore():
    """Returns a Chroma vector store connected to our persistent database."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def store_chunks(chunks, source: str):
    """Stores a document's chunks, replacing any older chunks from the same file."""
    vectorstore = get_vectorstore()

    try:
        vectorstore.delete(where={"source": source})
    except Exception:
        pass

    vectorstore.add_documents(chunks)
    return len(chunks)


def list_documents():
    """Returns the distinct documents currently stored in the vector database."""
    vectorstore = get_vectorstore()
    data = vectorstore.get()

    sources = set()
    for meta in (data.get("metadatas") or []):
        if meta and meta.get("source"):
            sources.add(meta["source"])

    documents = [
        {"source": source, "filename": os.path.basename(source)}
        for source in sorted(sources)
    ]
    return documents


def get_first_chunk(source: str):
    """Returns the opening chunk (lowest page number) of a specific document."""
    vectorstore = get_vectorstore()
    data = vectorstore.get(where={"source": source})

    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    if not docs:
        return None

    pairs = list(zip(docs, metas))
    pairs.sort(key=lambda pair: (pair[1] or {}).get("page", 0))

    content, meta = pairs[0]
    return SimpleDoc(content, meta or {})


def get_page_chunks(source: str, page_index: int):
    """Returns all stored chunks belonging to a specific page (0-indexed)
    of a specific document. Fetches all of the document's chunks and
    filters in Python rather than relying on a database-side compound
    filter, since Chroma's filter syntax varies across versions and this
    is far more reliable for a one-off lookup like this.
    """
    vectorstore = get_vectorstore()
    data = vectorstore.get(where={"source": source})

    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    matches = [
        (doc, meta) for doc, meta in zip(docs, metas)
        if meta and meta.get("page") == page_index
    ]

    if not matches:
        return None

    combined_text = "\n\n".join(doc for doc, meta in matches)
    return SimpleDoc(combined_text, matches[0][1])
