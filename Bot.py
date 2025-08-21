import os
import time
import uuid
import re
from typing import List, Dict, Tuple

import streamlit as st
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

# ───────────────────────────────────────────────────────────────────────────────
# Quick sanity checks (env vars and PDF path)
# ───────────────────────────────────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    # Fail fast with a clear message in the UI
    st.error("Faltan variables de entorno: OPENAI_API_KEY o PINECONE_API_KEY. Configúralas y reinicia la app.")
    st.stop()

# Resolve PDF path from env/secret (falls back to repo root file)
PDF_PATH = os.getenv("PDF_PATH", "Condiciones_generales.pdf")
if not os.path.exists(PDF_PATH):
    st.error(f"No se encontró el PDF en la ruta indicada: '{PDF_PATH}'. Verifica el path y nombre del archivo, súbelo al repo o define la variable `PDF_PATH` en Secrets.")
    st.stop()

# Optional but nice-to-have for "hybrid": BM25 over local chunks
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

# ───────────────────────────────────────────────────────────────────────────────
# CONFIG (edit if you like)
# ───────────────────────────────────────────────────────────────────────────────

INDEX_NAME = "test-six"                     # new index we will (auto) create
NAMESPACE  = "warranty-es"                  # keep warranty data separate

EMBED_MODEL = "text-embedding-3-small"      # 1536-dim
CHAT_MODEL  = "gpt-5-nano-2025-08-07"       # as requested

CHUNK_SIZE = 1400                           # between 1200–1500, we’ll use 1400
CHUNK_OVERLAP = 200
TOP_K = 5
DENSE_FETCH = 15                            # fetch more dense hits, then blend
ALPHA = 0.5                                 # hybrid weight: 0=dense-only, 1=bm25-only (we blend 50/50)
MIN_CONFIDENCE = 0.25                       # if combined score is too low -> gentle redirect

# ───────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL UTILS
# ───────────────────────────────────────────────────────────────────────────────
def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

def read_pdf(path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(path)
    pages = []
    for i, p in enumerate(reader.pages):
        txt = clean_text(p.extract_text())
        pages.append((i + 1, txt))
    return pages

def chunk_page_text(page_num: int, text: str, chunk_size: int, overlap: int) -> List[Dict]:
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        snippet = text[start:end]

        # try to end on a sentence boundary if possible (keeps chunks coherent)
        if end < L:
            last_dot = snippet.rfind(". ")
            if last_dot != -1 and last_dot > int(0.6 * len(snippet)):
                end = start + last_dot + 1
                snippet = text[start:end]

        chunks.append({
            "id": f"{page_num}-{start}-{uuid.uuid4().hex[:8]}",
            "page": page_num,
            "text": snippet
        })
        if end == L:
            break
        start = max(end - overlap, end)  # safety
    return chunks

def chunk_pdf(pages: List[Tuple[int, str]], size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[Dict]:
    all_chunks = []
    for page_num, txt in pages:
        if txt:
            all_chunks.extend(chunk_page_text(page_num, txt, size, overlap))
    return all_chunks

def normalize_scores(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = list(d.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: 1.0 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}

# ───────────────────────────────────────────────────────────────────────────────
# EXTERNAL SERVICES
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_clients():
    # Keys must come from env vars (no hardcoding)
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        st.error("Faltan las claves OPENAI_API_KEY y/o PINECONE_API_KEY en el entorno. Configúralas y recarga.")
        st.stop()
    return OpenAI(), Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def ensure_index(pc: Pinecone):
    # Handle different return shapes across pinecone client versions
    _list = pc.list_indexes()
    names = []
    for i in _list:
        if isinstance(i, dict) and "name" in i:
            names.append(i["name"])          # e.g., {"name": "idx"}
        elif hasattr(i, "name"):
            names.append(i.name)               # e.g., object with .name
        else:
            names.append(str(i))               # e.g., plain string
    if INDEX_NAME not in names:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # wait until ready
        while True:
            desc = pc.describe_index(INDEX_NAME)
            if desc.status["ready"]:
                break
            time.sleep(1.0)
    return pc.Index(INDEX_NAME)

def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    # batched embeddings
    out = []
    B = 256
    for i in range(0, len(texts), B):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i+B])
        out.extend([d.embedding for d in resp.data])
    return out

def upsert_chunks(index, client: OpenAI, chunks: List[Dict], namespace: str):
    B = 100
    for i in range(0, len(chunks), B):
        batch = chunks[i:i+B]
        embs = embed_texts(client, [c["text"] for c in batch])
        vectors = []
        for c, e in zip(batch, embs):
            vectors.append({
                "id": c["id"],
                "values": e,
                "metadata": {
                    "text": c["text"],
                    "page": c["page"],
                    "source": "Condiciones_generales.pdf",
                }
            })
        index.upsert(vectors=vectors, namespace=namespace)

def namespace_count(index, namespace: str) -> int:
    stats = index.describe_index_stats()
    ns = (stats or {}).get("namespaces", {})
    return int(ns.get(namespace, {}).get("vector_count", 0))

# ───────────────────────────────────────────────────────────────────────────────
# INGEST (runs once; cached)
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def ingest_pdf_build_bm25_and_index():
    """Reads the PDF, builds chunks, creates Pinecone index if needed,
    (re)upserts vectors for this namespace if empty, and builds BM25."""
    client, pc = get_clients()
    index = ensure_index(pc)

    pages = read_pdf(PDF_PATH)
    chunks = chunk_pdf(pages, CHUNK_SIZE, CHUNK_OVERLAP)

    # Upsert only if namespace empty
    if namespace_count(index, NAMESPACE) == 0:
        upsert_chunks(index, client, chunks, NAMESPACE)

    # BM25 for local hybrid (optional)
    bm25 = None
    tokenized = None
    if HAS_BM25:
        tokenized = [re.findall(r"\w+", c["text"].lower()) for c in chunks]
        bm25 = BM25Okapi(tokenized)

    # quick lookups
    id2chunk = {c["id"]: c for c in chunks}
    return index, bm25, tokenized, id2chunk

# ───────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ───────────────────────────────────────────────────────────────────────────────
def dense_search(index, client: OpenAI, query: str, top_k: int = DENSE_FETCH) -> Dict[str, float]:
    q_emb = embed_texts(client, [query])[0]
    res = index.query(
        namespace=NAMESPACE,
        vector=q_emb,
        top_k=top_k,
        include_metadata=True
    )
    scores = {}
    for m in res.get("matches", []):
        scores[m["id"]] = float(m["score"])
    return scores

def bm25_scores(bm25, tokens, query: str) -> Dict[str, float]:
    if not bm25:
        return {}
    q_tokens = re.findall(r"\w+", query.lower())
    arr = bm25.get_scores(q_tokens)
    # return as dict id->score (use same order as tokens/chunks)
    return {str(idx): float(s) for idx, s in enumerate(arr)}  # temp indices; we’ll map below

from typing import Tuple as _TupleAlias  # local alias to avoid confusion

def hybrid_retrieve(query: str, top_k: int, id2chunk: Dict[str, Dict], bm25, tokenized) -> _TupleAlias[List[Dict], float]:
    client, _ = get_clients()
    index, _, _, _ = ingest_pdf_build_bm25_and_index()

    dense = dense_search(index, client, query, DENSE_FETCH)  # id -> score
    ndense = normalize_scores(dense)

    nbm25: Dict[str, float] = {}
    if bm25 and tokenized:
        # produce id->score in terms of *actual* ids
        b = bm25.get_scores(re.findall(r"\w+", query.lower()))
        # map BM25 array indices back to real chunk ids by construction order
        # rebuild an ordered list of ids once
        ordered_ids = list(id2chunk.keys())
        raw = {ordered_ids[i]: float(b[i]) for i in range(len(ordered_ids))}
        nbm25 = normalize_scores(raw)

    combined_ids = set(ndense.keys()) | set(nbm25.keys())
    combined = {cid: ALPHA * nbm25.get(cid, 0.0) + (1 - ALPHA) * ndense.get(cid, 0.0)
                for cid in combined_ids}

    top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
    out = []
    for cid, score in top:
        c = id2chunk[cid]
        out.append({"id": cid, "score": score, "page": c["page"], "text": c["text"]})
    max_score = top[0][1] if top else 0.0
    return out, max_score

# ───────────────────────────────────────────────────────────────────────────────
# LLM ORCHESTRATION
# ───────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres el *Asistente de Garantías Commencal*.
Tu función es ayudar en **español de Chile** a clientes con preguntas sobre **garantías de bicicletas** usando la información provista en el documento adjunto (fragmentos con número de página).
Reglas:
- Sé educado, claro y conciso. Usa viñetas cuando ayuden.
- Puedes hacer **preguntas de seguimiento** para identificar modelo, año, tamaño de rueda, recorrido, etc., si es relevante.
- Si la pregunta **no es sobre garantías**: ofrece una **breve orientación general** (1–2 frases) y **redirígela** inmediatamente al ámbito de garantías.
- Si tu **confianza** en la información recuperada es baja, indica que **no encuentras** esa parte exacta en el documento y sugiere qué datos faltan o cómo reformular.
- **No inventes** políticas ni detalles técnicos que no aparezcan en el documento.
- **Se breve**, sin respuesta tediosas o muy largas. Tiene que ser fácil de leer y entender
- Puedes responder en cualquier idioma, mantén el foco en garantías.
"""

def build_context_block(snippets: List[Dict]) -> str:
    # Keep total context reasonably small
    pieces = []
    used = 0
    LIMIT = 4000  # characters
    for s in snippets:
        frag = f"[p.{s['page']}] {s['text']}"
        if used + len(frag) > LIMIT:
            break
        pieces.append(frag)
        used += len(frag)
    return "\n\n".join(pieces)

def call_llm(messages: List[Dict]) -> str:
    client, _ = get_clients()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )
    return resp.choices[0].message.content

def answer_query(query: str, history_for_llm: List[Dict]) -> Tuple[str, List[Dict], float]:
    index, bm25, tokenized, id2chunk = ingest_pdf_build_bm25_and_index()
    snippets, max_conf = hybrid_retrieve(query, TOP_K, id2chunk, bm25, tokenized)

    context_block = build_context_block(snippets)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history_for_llm)  # short chat history (user/assistant turns)
    messages.append({
        "role": "user",
        "content": f"Consulta del cliente:\n{query}\n\nFragmentos del documento (para consultar, con número de página):\n{context_block}\n\nConfianza aproximada: {round(max_conf, 3)}"
    })

    text = call_llm(messages)

    # If confidence very low, prepend a gentle redirect (safety net)
    if max_conf < MIN_CONFIDENCE:
        tip = ("No encuentro información precisa sobre eso en el documento de garantías. "
               "Puedo ayudarte con dudas de **garantías** (cobertura, condiciones, procesos). "
               "¿Podrías contarme el **modelo y año** de tu bicicleta y el **problema**?")
        text = f"{tip}\n\n{text}"

    return text, snippets, max_conf

# ───────────────────────────────────────────────────────────────────────────────
# UI (simple WhatsApp‑ish)
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Asistente de Garantías", page_icon="🚲", layout="centered")

st.markdown("""
<style>
/* WhatsApp‑ish bubbles */
.stChatMessage[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] p {
  line-height: 1.35;
}
.user-bubble {
  background: #d9fdd3; /* WhatsApp green bubble */
  padding: 10px 14px; border-radius: 16px; display: inline-block;
}
.bot-bubble {
  background: #ffffff; border: 1px solid #e6e6e6;
  padding: 10px 14px; border-radius: 16px; display: inline-block;
}
.small-gray { color:#6b7280; font-size:12px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuración")
    st.write("- **Modelo chat:** gpt-5-nano-2025-08-07")
    st.write("- **Embeddings:** text-embedding-3-small (1536)")
    st.write(f"- **Índice Pinecone:** `{INDEX_NAME}` (namespace: `{NAMESPACE}`)")
    st.write("- **Búsqueda:** híbrida (densa + BM25 local)")
    st.write(f"- **PDF:** {PDF_PATH}")
    st.divider()
    #st.caption("Si es la primera vez, el índice se creará y se ingestará automáticamente.\n"
     #          "Las claves deben venir de las variables de entorno OPENAI_API_KEY y PINECONE_API_KEY.\n"
      #         "Para ejecutar: `streamlit run Bot.py`.\n"
       #        "Paquete correcto: `pinecone` (no `pinecone-client`).")

    if st.button("🔁 Reingestar PDF en el índice"):
        # Clear caches so ingest runs again
        ingest_pdf_build_bm25_and_index.clear()
        st.success("Listo. Vuelve a hacer una pregunta (se reconstruyó el índice/estado cacheado).")

# warm up / ensure index exists & (if empty) ingest
_ = ingest_pdf_build_bm25_and_index()

st.markdown("### 💬 Asistente de Garantías Commencal (RAG)")
st.caption("Haz tu pregunta sobre garantías. Ej.: *¿Qué no cubre la garantía para el cuadro META 2021?*")

if "chat" not in st.session_state:
    st.session_state.chat = []

# Paint history
for role, content, src in st.session_state.chat:
    with st.chat_message(role):
        bubble_class = "user-bubble" if role == "user" else "bot-bubble"
        st.markdown(f"<div class='{bubble_class}'>{content}</div>", unsafe_allow_html=True)
        # Sources expander for assistant turns
        if role == "assistant" and src:
            with st.expander("📎 Fuentes (fragmentos y páginas)"):
                for i, s in enumerate(src, 1):
                    st.markdown(f"**{i}.** p.{s['page']} — {s['text'][:350]}{'…' if len(s['text'])>350 else ''}")

prompt = st.chat_input("Escribe un mensaje…")
if prompt:
    # Show user bubble immediately
    st.session_state.chat.append(("user", prompt, None))
    with st.chat_message("user"):
        st.markdown(f"<div class='user-bubble'>{prompt}</div>", unsafe_allow_html=True)

    # Build a compact history for the LLM (last few turns)
    history_for_llm = []
    for role, content, _ in st.session_state.chat[-6:]:
        if role in ("user", "assistant"):
            history_for_llm.append({"role": role, "content": content})

    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            reply, sources, conf = answer_query(prompt, history_for_llm)
            st.markdown(f"<div class='bot-bubble'>{reply}</div>", unsafe_allow_html=True)
            with st.expander("📎 Fuentes (fragmentos y páginas)"):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**{i}.** p.{s['page']} — {s['text'][:350]}{'…' if len(s['text'])>350 else ''}")
    st.session_state.chat.append(("assistant", reply, sources))