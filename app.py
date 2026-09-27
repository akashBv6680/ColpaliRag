import os
import time
import hashlib
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd

# PDF & image handling
import fitz  # PyMuPDF

# LangChain text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ChromaDB
import chromadb
from chromadb.config import Settings

# Gemini
from google import genai
from google.genai import types

# TTS
import nest_asyncio
from gtts import gTTS
from edge_tts import EdgeTTS

# AgentOps tracker (you will drop agentops_config.py in the same repo)
from agentops_config import tracker

# Allow async TTS inside Streamlit
nest_asyncio.apply()

# -----------------------------
# Config & constants
# -----------------------------

GEMINI_MODEL = "gemini-3.6-flash"  # exactly as you specified

CHROMA_PERSIST_DIR = "./chroma_colpali"
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection_name = "colpali_docs"
collection = chroma_client.get_or_create_collection(name=collection_name)

# Initialize Gemini client
gemini_client = genai.Client()

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# -----------------------------
# Helper utilities
# -----------------------------

def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()

def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    content = uploaded_file.read()
    if name.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df.to_string()
    if name.endswith(".json"):
        return content.decode("utf-8", errors="ignore")
    if name.endswith(".html") or name.endswith(".htm"):
        soup = BeautifulSoup(content, "lxml")
        return soup.get_text(separator="\n", strip=True)
    if name.endswith(".xml"):
        soup = BeautifulSoup(content, "lxml")
        return soup.get_text(separator="\n", strip=True)
    if name.endswith(".pdf"):
        # We'll handle PDF separately for images; here just basic text
        return extract_text_from_pdf_bytes(content)
    return content.decode("utf-8", errors="ignore")

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n\n".join(text_parts)

def render_pdf_to_images(pdf_bytes: bytes, max_pages: Optional[int] = None) -> List[bytes]:
    """
    Render each PDF page to PNG bytes.
    Returns a list of PNG byte blobs, one per page.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_images = []
    pages_to_process = len(doc) if max_pages is None else min(len(doc), max_pages)
    for i in range(pages_to_process):
        page = doc[i]
        # Render at 2x zoom for better quality
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        page_images.append(png_bytes)
        pix = None
    doc.close()
    return page_images

def fetch_url_text(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ColpaliRAG/1.0; +https://github.com/akashBv6680/ColpaliRag)"
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "lxml")
    return soup.get_text(separator="\n", strip=True)

# -----------------------------
# CAG-style cache (simple in-memory)
# -----------------------------

if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}  # key: hash(query + context_ids) -> answer

def get_cached_answer(key: str) -> Optional[str]:
    return st.session_state.answer_cache.get(key)

def set_cached_answer(key: str, answer: str):
    st.session_state.answer_cache[key] = answer

# -----------------------------
# RAG core functions
# -----------------------------

def add_document_to_chroma(
    doc_id: str,
    text: str,
    metadata: Dict[str, Any]
):
    """
    Split text into chunks and add to Chroma with basic text embeddings.
    For now, we use a simple placeholder embedding (zeros) and rely on
    Gemini for understanding; you can later plug in real embeddings.
    """
    chunks = text_splitter.split_text(text)
    if not chunks:
        return

    ids = []
    embeddings = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        # Placeholder embedding: 1536-dim zero vector (just to satisfy schema)
        # Later you can replace this with real embeddings (e.g., from an API).
        emb = [0.0] * 1536
        ids.append(chunk_id)
        embeddings.append(emb)
        meta = {
            "doc_id": doc_id,
            "chunk_index": i,
            **metadata
        }
        metadatas.append(meta)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=chunks
    )

def retrieve_chunks_for_query(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve chunks for a query.
    Since we use placeholder embeddings, this is just a demo retrieval.
    For a real system, plug in proper embeddings + similarity search.
    """
    # Placeholder query embedding
    query_emb = [0.0] * 1536
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    out = []
    for doc, meta in zip(docs, metas):
        out.append({"text": doc, "metadata": meta})
    return out

def ask_gemini_visual_rag(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    page_images: Optional[List[bytes]] = None,
    use_images: bool = True
) -> str:
    """
    Ask Gemini with:
      - text context from retrieved chunks
      - optional page images (PNG bytes) for visual understanding.
    """
    context_text = "\n\n---\n\n".join([c["text"] for c in retrieved_chunks])

    prompt = (
        "You are an expert assistant for a ColPali-style RAG system.\n"
        "Use the following retrieved context and (optionally) page images to answer the user's question.\n"
        "If images are provided, pay special attention to tables, charts, diagrams, and layout.\n\n"
        "Retrieved context:\n"
        f"{context_text}\n\n"
        f"User question: {query}\n\n"
        "Answer concisely and clearly. If the information is not in the context or images, say so."
    )

    contents = []

    # Add text prompt
    contents.append(prompt)

    # Add images if available and requested
    if use_images and page_images:
        # Gemini expects either a URL or inline bytes for images.
        # Here we use inline bytes via types.Part.from_bytes for images.
        for png_bytes in page_images:
            image_part = types.Part.from_bytes(
                data=png_bytes,
                mime_type="image/png"
            )
            contents.append(image_part)

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.3,
            top_p=0.8,
        ),
    )

    # Extract text from response
    answer = ""
    if response and response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.text:
                    answer += part.text
    return answer.strip()

# -----------------------------
# TTS functions
# -----------------------------

def text_to_speech_gtts(text: str, lang: str = "en") -> Optional[bytes]:
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            fp_path = fp.name
        with open(fp_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(fp_path)
        return audio_bytes
    except Exception as e:
        st.warning(f"gTTS failed: {e}")
        return None

async def text_to_speech_edge(text: str, lang: str = "en-US") -> Optional[bytes]:
    try:
        tts = EdgeTTS()
        # Choose a voice; you can customize
        voice = "en-US-JennyNeural" if "en" in lang else "en-US-JennyNeural"
        tts.set_voice(voice)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            await tts.save(fp.name, text)
            fp_path = fp.name
        with open(fp_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(fp_path)
        return audio_bytes
    except Exception as e:
        st.warning(f"Edge TTS failed: {e}")
        return None

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="ColPali RAG",
    page_icon="🧠",
    layout="wide"
)

st.title("ColPali-style Visual RAG with Gemini")

st.markdown(
    """
    This app demonstrates a **ColPali-inspired** RAG system:
    - Documents are split into text chunks for retrieval.
    - PDFs are also rendered as images.
    - When answering, **Gemini (gemini-3.6-flash)** receives both text context and page images,
      so it can understand tables, charts, and complex layouts.
    """
)

# Sidebar: document upload & settings
with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, CSV, HTML, XML, JSON)",
        type=["pdf", "txt", "csv", "html", "htm", "xml", "json"],
        accept_multiple_files=True
    )

    url_input = st.text_input("Or ingest a URL (e.g., https://example.com/article)")

    max_pdf_pages = st.number_input(
        "Max PDF pages to render as images (per doc)",
        min_value=1,
        max_value=50,
        value=10,
        help="Limit images per PDF to control cost and latency."
    )

    use_images_in_answer = st.checkbox(
        "Use page images in answers (visual RAG)",
        value=True,
        help="If unchecked, only text context is sent to Gemini."
    )

    st.divider()

    st.header("Answer settings")

    enable_tts = st.checkbox("Enable text-to-speech for answers", value=True)
    tts_engine = st.selectbox(
        "TTS engine",
        options=["gTTS", "Edge TTS"],
        index=0
    )
    tts_lang = st.text_input("TTS language code", value="en")

    st.divider()

    if st.button("Clear answer cache"):
        st.session_state.answer_cache = {}
        st.success("Answer cache cleared.")

# Process uploaded documents
if uploaded_files:
    for uploaded_file in uploaded_files:
        doc_name = uploaded_file.name
        doc_id = hash_content(f"{doc_name}_{time.time()}")

        # Track upload
        try:
            tracker.track_document_upload(
                filename=doc_name,
                size=uploaded_file.size,
                chunk_count=0  # we don't know yet; can be improved
            )
        except Exception:
            pass

        if doc_name.lower().endswith(".pdf"):
            # Text
            pdf_bytes = uploaded_file.read()
            text_content = extract_text_from_pdf_bytes(pdf_bytes)

            # Images
            page_images = render_pdf_to_images(pdf_bytes, max_pages=max_pdf_pages)

            # Store images in session for later retrieval by doc_id
            if "pdf_images" not in st.session_state:
                st.session_state.pdf_images = {}
            st.session_state.pdf_images[doc_id] = page_images

            # Add text to Chroma
            add_document_to_chroma(
                doc_id=doc_id,
                text=text_content,
                metadata={
                    "filename": doc_name,
                    "source_type": "pdf",
                    "has_images": True,
                    "num_pages": len(page_images)
                }
            )
        else:
            # Non-PDF: just text
            text_content = extract_text_from_file(uploaded_file)
            add_document_to_chroma(
                doc_id=doc_id,
                text=text_content,
                metadata={
                    "filename": doc_name,
                    "source_type": "file",
                    "has_images": False,
                    "num_pages": 0
                }
            )

    st.sidebar.success(f"Processed {len(uploaded_files)} file(s).")

# Process URL
if url_input:
    try:
        url_text = fetch_url_text(url_input)
        doc_id = hash_content(f"url_{url_input}_{time.time()}")
        add_document_to_chroma(
            doc_id=doc_id,
            text=url_text,
            metadata={
                "filename": url_input,
                "source_type": "url",
                "has_images": False,
                "num_pages": 0
            }
        )
        st.sidebar.success("URL ingested successfully.")
        url_input = ""  # clear input
    except Exception as e:
        st.sidebar.error(f"Failed to ingest URL: {e}")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask anything about your documents...")

if user_query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve chunks
    retrieved_chunks = retrieve_chunks_for_query(user_query, top_k=5)

    # Collect relevant page images (from all docs for simplicity; can be refined)
    page_images_to_send = None
    if use_images_in_answer and "pdf_images" in st.session_state:
        all_images = []
        for doc_id, imgs in st.session_state.pdf_images.items():
            all_images.extend(imgs)
        # Limit number of images sent to Gemini to control cost
        max_images_to_send = 5
        page_images_to_send = all_images[:max_images_to_send]

    # Prepare cache key
    context_ids = "|".join(
        [f"{c['metadata'].get('doc_id')}_{c['metadata'].get('chunk_index')}" for c in retrieved_chunks]
    )
    cache_key = hash_content(f"{user_query}||{context_ids}||{use_images_in_answer}")

    answer = get_cached_answer(cache_key)

    if answer is None:
        with st.chat_message("assistant"):
            with st.spinner("Thinking (ColPali-style visual RAG)..."):
                start_time = time.time()
                try:
                    answer = ask_gemini_visual_rag(
                        query=user_query,
                        retrieved_chunks=retrieved_chunks,
                        page_images=page_images_to_send,
                        use_images=use_images_in_answer
                    )
                except Exception as e:
                    answer = f"Error while generating answer: {e}"

                response_time = time.time() - start_time

                # Track query
                try:
                    tracker.track_rag_query(
                        query=user_query,
                        retrieved_chunks=retrieved_chunks,
                        answer_length=len(answer),
                        response_time=response_time
                    )
                except Exception:
                    pass

                set_cached_answer(cache_key, answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

            # TTS
            if enable_tts and answer:
                try:
                    if tts_engine == "gTTS":
                        audio_bytes = text_to_speech_gtts(answer, lang=tts_lang)
                    else:
                        audio_bytes = asyncio.run(
                            text_to_speech_edge(answer, lang=tts_lang or "en-US")
                        )
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mpeg", start_time=0)
                        try:
                            tracker.track_tts_generation(
                                text_length=len(answer),
                                language=tts_lang or "en",
                                engine=tts_engine,
                                audio_duration=0  # approximate if needed
                            )
                        except Exception:
                            pass
                except Exception as e:
                    st.warning(f"TTS error: {e}")
    else:
        # Use cached answer
        with st.chat_message("assistant"):
            st.markdown(f"*[cached]* {answer}")
        st.session_state.messages.append({"role": "assistant", "content": f"[cached] {answer}"})

# Footer: AgentOps link
st.divider()
if tracker and hasattr(tracker, "get_session_dashboard_url"):
    dashboard_url = tracker.get_session_dashboard_url()
    if dashboard_url:
        st.markdown(f"**AgentOps dashboard:** [{dashboard_url}]({dashboard_url})")
