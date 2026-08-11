import json
import re

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL_NAME
from app.services.vector_store import get_vectorstore, get_first_chunk, get_page_chunks
from app.services.bm25_service import bm25_search
from app.services.reranker_service import rerank
from app.services.web_search_service import search_web

_groq_client = None

DOC_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a specific "
    "document. Use ONLY the context provided with each question to answer. "
    "If the answer is not in the context, say you don't know."
)

WEB_SYSTEM_PROMPT = (
    "You are a helpful assistant. The user's question could not be answered "
    "from their uploaded document, so you have been given live web search "
    "results instead. Answer clearly and concisely using these results."
)

RERANK_THRESHOLD = 2.0
SOURCE_DELIMITER = "\n@@SOURCES@@\n"


def get_groq_client():
    global _groq_client

    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def hybrid_retrieve(question: str, document: str, k: int = 10):
    """Combines semantic search and BM25 keyword search into one candidate
    pool, then reranks that pool with a cross-encoder for a more precise
    final ordering than either search alone would give.
    """
    vectorstore = get_vectorstore()
    filter_dict = {"source": document} if document else None

    try:
        if filter_dict:
            semantic_results = vectorstore.similarity_search(question, k=k, filter=filter_dict)
        else:
            semantic_results = vectorstore.similarity_search(question, k=k)
    except Exception:
        semantic_results = []

    keyword_results = bm25_search(question, document=document, k=k)

    seen = set()
    candidates = []
    for doc in semantic_results + keyword_results:
        key = (doc.metadata.get("page"), doc.page_content[:80])
        if key not in seen:
            seen.add(key)
            candidates.append(doc)

    reranked = rerank(question, candidates)
    return reranked


def rewrite_query(question: str, history: list) -> str:
    """Asks the LLM to turn a possibly vague question into a clearer,
    standalone search query, using recent conversation for context.
    """
    history_text = "\n".join(
        f"Q: {t.get('question','')}\nA: {t.get('answer','')}" for t in (history or [])[-2:]
    )
    prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        "Rewrite the question below into a clear, specific, standalone "
        "search query. Return ONLY the rewritten query, nothing else.\n\n"
        f"Question: {question}"
    )
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else question
    except Exception:
        return question


def verify_citations(answer: str, sources: list) -> list:
    """Checks each candidate source against the generated answer and keeps
    only the ones that genuinely support a claim in it, so citations never
    point to unrelated content. Fails open (keeps sources) if the check
    itself errors out, rather than silently hiding everything.
    """
    if not sources:
        return sources

    numbered = "\n\n".join(f"[{i}] {s['snippet']}" for i, s in enumerate(sources))
    prompt = (
        "Below is an AI-generated answer, followed by numbered source "
        "excerpts. Return ONLY a JSON array of the numbers whose excerpt "
        "genuinely supports a claim made in the answer. If none do, "
        "return [].\n\n"
        f"Answer:\n{answer}\n\nSource excerpts:\n{numbered}"
    )
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r"\[[\d,\s]*\]", text)
        if match:
            indices = json.loads(match.group(0))
            verified = [s for i, s in enumerate(sources) if i in indices]
            return verified if verified else sources
    except Exception:
        pass

    return sources


def build_doc_messages(question, context, history):
    messages = [{"role": "system", "content": DOC_SYSTEM_PROMPT}]
    for turn in (history or [])[-3:]:
        messages.append({"role": "user", "content": turn.get("question", "")})
        messages.append({"role": "assistant", "content": turn.get("answer", "")})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    return messages


def build_web_messages(question, web_context, history):
    messages = [{"role": "system", "content": WEB_SYSTEM_PROMPT}]
    for turn in (history or [])[-3:]:
        messages.append({"role": "user", "content": turn.get("question", "")})
        messages.append({"role": "assistant", "content": turn.get("answer", "")})
    messages.append({"role": "user", "content": f"Web search results:\n{web_context}\n\nQuestion: {question}"})
    return messages


GENERATION_ERROR_MESSAGE = "I ran into a problem generating a response. Please try again in a moment."
NO_ANSWER_MESSAGE = "I couldn't find this in your document, and the web search didn't turn up anything reliable either."


def _get_document_chunks(question, document, history):
    """Hybrid retrieve + rerank; if the best match is still weak, rewrite
    the query once and retry before giving up on the document entirely.
    """
    reranked = hybrid_retrieve(question, document)
    confident = [doc for doc, score in reranked if score > RERANK_THRESHOLD]

    if not confident:
        rewritten = rewrite_query(question, history)
        if rewritten.strip().lower() != question.strip().lower():
            reranked = hybrid_retrieve(rewritten, document)
            confident = [doc for doc, score in reranked if score > RERANK_THRESHOLD]

    confident = confident[:3]

    intro_chunk = get_first_chunk(document)
    if intro_chunk is not None and confident:
        already_included = any(
            c.metadata.get("page") == intro_chunk.metadata.get("page") for c in confident
        )
        if not already_included:
            confident.append(intro_chunk)

    return confident[:4]


def _prepare(question, history, document):
    chunks = _get_document_chunks(question, document, history) if document else []

    if chunks:
        context = "\n\n".join(c.page_content for c in chunks)
        sources = [
            {"page": c.metadata.get("page", 0) + 1, "snippet": c.page_content[:300]}
            for c in chunks
        ]
        messages = build_doc_messages(question, context, history)
        return messages, sources, "document"

    web_results = search_web(question)
    if not web_results:
        return None, [], "none"

    web_context = "\n\n".join(f"{r['title']}\n{r['content']}\nSource: {r['url']}" for r in web_results)
    sources = [{"title": r["title"], "url": r["url"]} for r in web_results]
    messages = build_web_messages(question, web_context, history)
    return messages, sources, "web"


def answer_question(question: str, history: list = None, document: str = None) -> dict:
    messages, sources, source_type = _prepare(question, history, document)

    if messages is None:
        return {"answer": NO_ANSWER_MESSAGE, "sources": [], "source_type": "none"}

    client = get_groq_client()
    try:
        response = client.chat.completions.create(model=GROQ_MODEL_NAME, messages=messages, temperature=0.2)
        answer = response.choices[0].message.content
    except Exception:
        return {"answer": GENERATION_ERROR_MESSAGE, "sources": [], "source_type": "none"}

    if source_type == "document":
        sources = verify_citations(answer, sources)

    return {"answer": answer, "sources": sources, "source_type": source_type}


def stream_answer(question: str, history: list = None, document: str = None):
    messages, sources, source_type = _prepare(question, history, document)

    if messages is None:
        yield NO_ANSWER_MESSAGE
        yield SOURCE_DELIMITER + json.dumps({"type": "none", "sources": []})
        return

    client = get_groq_client()
    full_answer = ""
    try:
        stream = client.chat.completions.create(model=GROQ_MODEL_NAME, messages=messages, temperature=0.2, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_answer += delta
                yield delta
    except Exception:
        yield GENERATION_ERROR_MESSAGE

    if source_type == "document" and full_answer:
        sources = verify_citations(full_answer, sources)

    yield SOURCE_DELIMITER + json.dumps({"type": source_type, "sources": sources})
