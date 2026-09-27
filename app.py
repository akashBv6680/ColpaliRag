# app.py
# 📚 ColPali-Inspired Visual RAG AI Agent with Multilingual & Voice Support 🎙️
# Modules preserved:
# 1. Document Loader
# 2. RAG Chatbot
# 3. TTS Demo (Standalone)
#
# Added:
# - PDF page rendering to images
# - Visual page descriptions created with Gemini
# - Visual-description retrieval through ChromaDB
# - Relevant PDF page image sent directly to Gemini for final answers
#
# Note:
# This is a ColPali-inspired visual RAG implementation.
# It does not run the original ColPali multi-vector model locally.

import asyncio
import base64
import datetime
import hashlib
import io
import os
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import fitz
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentops_config import tracker


# -----------------------------------------------------------
# SQLite compatibility for Streamlit Cloud / ChromaDB
# -----------------------------------------------------------

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except ImportError:
    pass


# -----------------------------------------------------------
# Optional embedding model
# -----------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# -----------------------------------------------------------
# Gemini SDK
# -----------------------------------------------------------

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
except ImportError:
    genai = None
    types = None
    APIError = Exception


# -----------------------------------------------------------
# Optional Text-to-Speech engines
# -----------------------------------------------------------

try:
    import edge_tts
except Exception:
    edge_tts = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None


# -----------------------------------------------------------
# Streamlit page configuration
# -----------------------------------------------------------

st.set_page_config(
    page_title="ColPali Visual RAG AI Agent",
    page_icon="📚",
    layout="wide",
)


# -----------------------------------------------------------
# Application configuration
# -----------------------------------------------------------

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.6-flash"

COLLECTION_NAME = "uploaded_documents_colpali_rag"
CACHE_EXPIRY_SECONDS = 300

TEXT_CHUNK_SIZE = 500
TEXT_CHUNK_OVERLAP = 100

DEFAULT_RETRIEVAL_COUNT = 6
DEFAULT_MAX_VISUAL_PAGES = 4
DEFAULT_PDF_PAGES_TO_PROCESS = 10

LANGUAGE_DICT = {
    "English": "en",
    "Spanish": "es",
    "Arabic": "ar",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Tamil": "ta",
    "Bengali": "bn",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Chinese (Simplified)": "zh-Hans",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Turkish": "tr",
}


# -----------------------------------------------------------
# Session state
# -----------------------------------------------------------

def initialize_session_state() -> None:
    defaults = {
        "ingested_files": [],
        "cache": {},
        "messages_rag": [],
        "page_images": {},
        "processed_file_hashes": set(),
        "visual_index_count": 0,
        "agentops_initialized": False,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


initialize_session_state()

if not st.session_state.agentops_initialized:
    tracker.initialize()
    st.session_state.agentops_initialized = True


# -----------------------------------------------------------
# Cached application resources
# -----------------------------------------------------------

@st.cache_resource(show_spinner=False)
def initialize_rag_dependencies():
    if SentenceTransformer is None:
        return None, None, None

    database_path = os.path.join(
        tempfile.gettempdir(),
        "chroma_colpali_visual_rag",
    )
    os.makedirs(database_path, exist_ok=True)

    database_client = chromadb.PersistentClient(path=database_path)

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu",
    )

    gemini_client = None
    if GEMINI_API_KEY and genai:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    return database_client, embedding_model, gemini_client


@st.cache_resource(show_spinner=False)
def initialize_gemini_client():
    if GEMINI_API_KEY and genai:
        return genai.Client(api_key=GEMINI_API_KEY)

    return None


# -----------------------------------------------------------
# ChromaDB functions
# -----------------------------------------------------------

def get_collection():
    if "db_client" not in st.session_state:
        (
            st.session_state.db_client,
            st.session_state.model,
            st.session_state.gemini_client,
        ) = initialize_rag_dependencies()

    if st.session_state.db_client is None:
        return None

    try:
        return st.session_state.db_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Text and visual page retrieval for RAG"},
        )
    except Exception as error:
        st.error(f"Error accessing ChromaDB: {error}")
        return None


def get_embedding_model():
    if "model" not in st.session_state:
        (
            st.session_state.db_client,
            st.session_state.model,
            st.session_state.gemini_client,
        ) = initialize_rag_dependencies()

    return st.session_state.model


def create_embeddings(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()

    if model is None:
        raise RuntimeError(
            "SentenceTransformer is unavailable. "
            "Check that sentence-transformers is in requirements.txt."
        )

    vectors = model.encode(
        texts,
        convert_to_tensor=False,
        show_progress_bar=False,
    )

    return vectors.tolist()


def split_documents(
    text_data: str,
    chunk_size: int = TEXT_CHUNK_SIZE,
    chunk_overlap: int = TEXT_CHUNK_OVERLAP,
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    return splitter.split_text(text_data)


def store_records(
    documents: List[str],
    metadatas: List[Dict[str, Any]],
) -> int:
    if not documents:
        return 0

    collection = get_collection()

    if collection is None:
        return 0

    embeddings = create_embeddings(documents)
    ids = [str(uuid.uuid4()) for _ in documents]

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    return len(documents)


def retrieve_records(
    query: str,
    n_results: int = DEFAULT_RETRIEVAL_COUNT,
) -> List[Dict[str, Any]]:
    collection = get_collection()

    if collection is None or collection.count() == 0:
        return []

    query_embedding = create_embeddings([query])[0]
    result_count = min(n_results, collection.count())

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )

    records = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None

        records.append(
            {
                "document": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return records


# -----------------------------------------------------------
# PDF and document extraction
# -----------------------------------------------------------

def create_document_id(file_name: str, file_bytes: bytes) -> str:
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:20]
    clean_name = "".join(
        character if character.isalnum() else "_"
        for character in file_name
    )

    return f"{clean_name}_{file_hash}"


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    extracted_pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
        for page_number, page in enumerate(pdf_document, start=1):
            page_text = page.get_text("text").strip()

            if page_text:
                extracted_pages.append(
                    f"[PDF Page {page_number}]\n{page_text}"
                )

    return "\n\n".join(extracted_pages)


def render_pdf_pages(
    pdf_bytes: bytes,
    max_pages: int,
    zoom: float = 1.7,
) -> List[Tuple[int, bytes]]:
    page_images: List[Tuple[int, bytes]] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
        page_limit = min(pdf_document.page_count, max_pages)

        for page_index in range(page_limit):
            page = pdf_document.load_page(page_index)

            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom),
                alpha=False,
            )

            page_images.append(
                (
                    page_index + 1,
                    pixmap.tobytes("png"),
                )
            )

    return page_images


def extract_text_from_upload(
    uploaded_file,
) -> Tuple[str, bool]:
    file_name = uploaded_file.name.lower()
    file_type = uploaded_file.type or ""
    raw_bytes = uploaded_file.getvalue()

    try:
        if file_name.endswith(".pdf") or file_type == "application/pdf":
            return extract_text_from_pdf(raw_bytes), True

        if file_name.endswith(".csv"):
            dataframe = pd.read_csv(io.BytesIO(raw_bytes))
            return dataframe.to_string(index=False), True

        if file_name.endswith(".json"):
            return raw_bytes.decode("utf-8", errors="ignore"), True

        if file_name.endswith((".html", ".htm", ".xml")):
            soup = BeautifulSoup(raw_bytes, "lxml")
            return soup.get_text(separator="\n", strip=True), True

        return raw_bytes.decode("utf-8", errors="ignore"), True

    except Exception as error:
        return f"Error reading file: {error}", False


# -----------------------------------------------------------
# URL ingestion
# -----------------------------------------------------------

def load_text_from_url(url: str) -> Tuple[str, bool]:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(compatible; ColPaliVisualRAG/1.0)"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "lxml")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        page_text = soup.get_text(
            separator="\n",
            strip=True,
        )

        if not page_text:
            return "No readable text found at this URL.", False

        return page_text, True

    except Exception as error:
        return f"Error fetching URL {url}: {error}", False


# -----------------------------------------------------------
# Gemini visual analysis
# -----------------------------------------------------------

def analyze_pdf_page_with_gemini(
    image_bytes: bytes,
    page_number: int,
    file_name: str,
) -> str:
    """
    Produces a searchable visual description of one PDF page.

    This description is embedded in ChromaDB. The real page PNG is
    stored separately in session state and can be returned at query time.
    """

    gemini_client = initialize_gemini_client()

    if gemini_client is None or types is None:
        return (
            f"PDF page {page_number} from {file_name}. "
            "Visual analysis unavailable because Gemini is not configured."
        )

    prompt = f"""
Analyze this PDF page for visual retrieval.

File: {file_name}
Page: {page_number}

Describe all important visual and structural content in a retrieval-friendly format:
- document title and section headings
- tables: names, columns, rows, values, relationships
- charts and graphs: chart type, axes, trends, key values
- architecture diagrams: components, hierarchy, arrows, relationships
- images, icons, labels, and callouts
- page layout and key text visible in the visual

Do not invent information.
Return concise but information-rich plain text that can later be used to retrieve this exact page.
""".strip()

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1200,
            ),
        )

        if response.text:
            return response.text.strip()

    except Exception as error:
        return (
            f"Visual-analysis fallback for PDF page {page_number}: "
            f"Gemini analysis failed: {error}"
        )

    return f"PDF page {page_number} from {file_name}."


# -----------------------------------------------------------
# Document ingestion: text + visual page index
# -----------------------------------------------------------

def process_pdf_for_visual_rag(
    uploaded_file,
    max_pages_to_render: int,
) -> Tuple[int, int, str]:
    """
    Ingests:
    1. Text chunks for classic RAG.
    2. Visual page descriptions for image/page retrieval.
    3. Original page PNG bytes stored in session state.
    """

    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    document_id = create_document_id(file_name, file_bytes)

    raw_text = extract_text_from_pdf(file_bytes)
    text_chunks = split_documents(raw_text)

    text_metadatas = [
        {
            "source_name": file_name,
            "document_id": document_id,
            "record_type": "text",
            "page_number": 0,
        }
        for _ in text_chunks
    ]

    text_count = store_records(
        documents=text_chunks,
        metadatas=text_metadatas,
    )

    page_images = render_pdf_pages(
        pdf_bytes=file_bytes,
        max_pages=max_pages_to_render,
    )

    visual_documents = []
    visual_metadatas = []

    if document_id not in st.session_state.page_images:
        st.session_state.page_images[document_id] = {}

    for page_number, image_bytes in page_images:
        st.session_state.page_images[document_id][page_number] = image_bytes

        visual_description = analyze_pdf_page_with_gemini(
            image_bytes=image_bytes,
            page_number=page_number,
            file_name=file_name,
        )

        visual_record = (
            f"VISUAL PAGE DESCRIPTION\n"
            f"File: {file_name}\n"
            f"Page Number: {page_number}\n\n"
            f"{visual_description}"
        )

        visual_documents.append(visual_record)
        visual_metadatas.append(
            {
                "source_name": file_name,
                "document_id": document_id,
                "record_type": "visual_page",
                "page_number": page_number,
            }
        )

    visual_count = store_records(
        documents=visual_documents,
        metadatas=visual_metadatas,
    )

    return text_count, visual_count, document_id


def process_text_document(
    uploaded_file,
) -> Tuple[int, str]:
    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    document_id = create_document_id(file_name, file_bytes)

    raw_text, success = extract_text_from_upload(uploaded_file)

    if not success:
        raise RuntimeError(raw_text)

    text_chunks = split_documents(raw_text)

    metadatas = [
        {
            "source_name": file_name,
            "document_id": document_id,
            "record_type": "text",
            "page_number": 0,
        }
        for _ in text_chunks
    ]

    count = store_records(
        documents=text_chunks,
        metadatas=metadatas,
    )

    return count, document_id


def process_url_document(url: str) -> int:
    page_text, success = load_text_from_url(url)

    if not success:
        raise RuntimeError(page_text)

    document_id = "url_" + hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:20]

    chunks = split_documents(page_text)

    metadatas = [
        {
            "source_name": url,
            "document_id": document_id,
            "record_type": "url_text",
            "page_number": 0,
        }
        for _ in chunks
    ]

    return store_records(
        documents=chunks,
        metadatas=metadatas,
    )


# -----------------------------------------------------------
# Visual-image retrieval
# -----------------------------------------------------------

def get_retrieved_page_images(
    retrieved_records: List[Dict[str, Any]],
    maximum_images: int,
) -> List[Dict[str, Any]]:
    """
    Finds visual_page records from Chroma retrieval results,
    then loads their matching original PNG images.
    """

    selected_images = []
    already_selected = set()

    for record in retrieved_records:
        metadata = record.get("metadata", {})

        if metadata.get("record_type") != "visual_page":
            continue

        document_id = metadata.get("document_id")
        page_number = metadata.get("page_number")
        source_name = metadata.get("source_name", "Unknown file")

        page_key = (document_id, page_number)

        if page_key in already_selected:
            continue

        image_bytes = (
            st.session_state.page_images
            .get(document_id, {})
            .get(page_number)
        )

        if image_bytes:
            selected_images.append(
                {
                    "image_bytes": image_bytes,
                    "document_id": document_id,
                    "page_number": page_number,
                    "source_name": source_name,
                    "visual_description": record.get("document", ""),
                }
            )

            already_selected.add(page_key)

        if len(selected_images) >= maximum_images:
            break

    return selected_images


def build_text_context(
    retrieved_records: List[Dict[str, Any]],
) -> str:
    context_parts = []

    for index, record in enumerate(retrieved_records, start=1):
        metadata = record.get("metadata", {})
        record_type = metadata.get("record_type", "text")
        source_name = metadata.get("source_name", "Unknown")
        page_number = metadata.get("page_number", 0)

        if record_type == "visual_page":
            label = (
                f"Visual page description from {source_name}, "
                f"page {page_number}"
            )
        else:
            label = f"Text context from {source_name}"

        context_parts.append(
            f"[Source {index}: {label}]\n"
            f"{record.get('document', '')}"
        )

    if not context_parts:
        return "No relevant documents were retrieved."

    return "\n\n---\n\n".join(context_parts)


# -----------------------------------------------------------
# Gemini response generation with retrieved images
# -----------------------------------------------------------

def call_gemini_visual_rag(
    query: str,
    text_context: str,
    retrieved_images: List[Dict[str, Any]],
    selected_language: str,
    max_retries: int = 3,
) -> Dict[str, str]:
    gemini_client = initialize_gemini_client()

    if gemini_client is None or types is None:
        return {
            "error": (
                "Gemini client is not configured. "
                "Add GEMINI_API_KEY to Streamlit secrets."
            )
        }

    system_instruction = f"""
You are an expert multimodal RAG assistant.

Use only the supplied text context and the retrieved document page images to answer the user question.

Rules:
1. Inspect images carefully for diagrams, charts, tables, labels, arrows, and visual relationships.
2. Use the visual page only when it is relevant to the question.
3. Do not invent data that is absent from the retrieved context or images.
4. If the answer is not supported, clearly say that.
5. Cite the source file and PDF page number when possible.
6. Respond clearly and accurately in {selected_language}.
""".strip()

    prompt = f"""
RETRIEVED DOCUMENT CONTEXT:
{text_context}

USER QUESTION:
{query}
""".strip()

    contents: List[Any] = [prompt]

    for retrieved_image in retrieved_images:
        source_name = retrieved_image["source_name"]
        page_number = retrieved_image["page_number"]

        contents.append(
            f"Retrieved visual source: {source_name}, PDF page {page_number}."
        )

        contents.append(
            types.Part.from_bytes(
                data=retrieved_image["image_bytes"],
                mime_type="image/png",
            )
        )

    retry_delay = 1

    for attempt in range(max_retries):
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                top_p=0.8,
                max_output_tokens=2048,
            )

            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )

            response_text = getattr(response, "text", None)

            if response_text:
                return {"response": response_text.strip()}

            return {"error": "Gemini returned an empty response."}

        except APIError as error:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return {"error": str(error)}

        except Exception as error:
            return {"error": str(error)}

    return {"error": "Gemini request failed after retries."}


# -----------------------------------------------------------
# CAG-enhanced RAG pipeline
# -----------------------------------------------------------

def create_cache_key(
    query: str,
    selected_language: str,
    max_visual_images: int,
) -> str:
    source = (
        f"{query}|{selected_language}|"
        f"{max_visual_images}|{GEMINI_MODEL}"
    )

    return hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()


def rag_pipeline(
    query: str,
    selected_language: str,
    retrieval_count: int,
    maximum_visual_images: int,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    cache_key = create_cache_key(
        query=query,
        selected_language=selected_language,
        max_visual_images=maximum_visual_images,
    )

    cache = st.session_state.cache

    if (
        cache_key in cache
        and time.time() - cache[cache_key]["timestamp"]
        < CACHE_EXPIRY_SECONDS
    ):
        cached = cache[cache_key]

        return (
            cached["answer"],
            cached["retrieved_records"],
            cached["retrieved_images"],
            True,
        )

    retrieved_records = retrieve_records(
        query=query,
        n_results=retrieval_count,
    )

    text_context = build_text_context(retrieved_records)

    retrieved_images = get_retrieved_page_images(
        retrieved_records=retrieved_records,
        maximum_images=maximum_visual_images,
    )

    response_json = call_gemini_visual_rag(
        query=query,
        text_context=text_context,
        retrieved_images=retrieved_images,
        selected_language=selected_language,
    )

    if "error" in response_json:
        answer = f"Generation error: {response_json['error']}"
    else:
        answer = response_json.get(
            "response",
            "No response text.",
        )

    st.session_state.cache[cache_key] = {
        "answer": answer,
        "timestamp": time.time(),
        "retrieved_records": retrieved_records,
        "retrieved_images": retrieved_images,
    }

    return answer, retrieved_records, retrieved_images, False


# -----------------------------------------------------------
# Text-to-speech utilities
# -----------------------------------------------------------

async def edge_tts_async(
    text: str,
    voice: str,
    rate: str = "+0%",
):
    if edge_tts is None:
        return None

    communication = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
    )

    audio_buffer = io.BytesIO()

    async for chunk in communication.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])

    return audio_buffer.getvalue()


def tts_edge(
    text: str,
    voice: str = "en-US-AriaNeural",
):
    try:
        return asyncio.run(
            edge_tts_async(
                text=text,
                voice=voice,
            )
        ), None
    except Exception as error:
        return None, str(error)


def tts_gtts(
    text: str,
    language_code: str = "en",
):
    if gTTS is None:
        return None, "gTTS is not available."

    try:
        audio_buffer = io.BytesIO()

        gTTS(
            text=text,
            lang=language_code,
        ).write_to_fp(audio_buffer)

        return audio_buffer.getvalue(), None

    except Exception as error:
        return None, str(error)


def synthesize(
    text: str,
    engine: str,
    language_code: str,
):
    edge_voice_map = {
        "en": "en-US-AriaNeural",
        "hi": "hi-IN-SwaraNeural",
        "ta": "ta-IN-PallaviNeural",
        "bn": "bn-IN-TanishaaNeural",
        "es": "es-ES-ElviraNeural",
        "fr": "fr-FR-DeniseNeural",
        "de": "de-DE-KatjaNeural",
        "ar": "ar-SA-ZariyahNeural",
        "zh-Hans": "zh-CN-XiaoxiaoNeural",
        "zh-cn": "zh-CN-XiaoxiaoNeural",
        "ja": "ja-JP-NanamiNeural",
        "ko": "ko-KR-SunHiNeural",
        "pt": "pt-PT-RaquelNeural",
        "it": "it-IT-ElsaNeural",
        "nl": "nl-NL-ColetteNeural",
        "tr": "tr-TR-EmelNeural",
        "ru": "ru-RU-SvetlanaNeural",
    }

    if engine == "Edge-TTS":
        selected_voice = edge_voice_map.get(
            language_code,
            "en-US-AriaNeural",
        )

        audio, error = tts_edge(
            text=text,
            voice=selected_voice,
        )

        if audio:
            return audio, "audio/mp3", None

        fallback_audio, fallback_error = tts_gtts(
            text=text,
            language_code=language_code,
        )

        return (
            fallback_audio,
            "audio/mp3",
            error or fallback_error,
        )

    if engine == "gTTS":
        audio, error = tts_gtts(
            text=text,
            language_code=language_code,
        )

        return audio, "audio/mp3", error

    return None, None, "Unknown TTS engine."


# -----------------------------------------------------------
# Clear storage
# -----------------------------------------------------------

def clear_rag_storage():
    if "db_client" not in st.session_state:
        (
            st.session_state.db_client,
            st.session_state.model,
            st.session_state.gemini_client,
        ) = initialize_rag_dependencies()

    database_client = st.session_state.db_client

    if database_client:
        try:
            database_client.delete_collection(
                name=COLLECTION_NAME,
            )
        except Exception:
            pass

    st.session_state.ingested_files = []
    st.session_state.cache = {}
    st.session_state.messages_rag = []
    st.session_state.page_images = {}
    st.session_state.processed_file_hashes = set()
    st.session_state.visual_index_count = 0

    st.rerun()


# -----------------------------------------------------------
# Sidebar
# -----------------------------------------------------------

st.sidebar.title("RAG Settings ⚙️")

menu = st.sidebar.radio(
    "Select Module",
    [
        "Document Loader",
        "RAG Chatbot",
        "TTS Demo (Standalone)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AgentOps Monitoring")

if tracker.is_initialized:
    session_display = (
        tracker.session_id[:8] + "..."
        if tracker.session_id
        else "active"
    )
    st.sidebar.success(f"✅ AgentOps session: {session_display}")

    dashboard_url = tracker.get_session_dashboard_url()

    if dashboard_url:
        st.sidebar.markdown(
            f"[📊 View Dashboard]({dashboard_url})"
        )
else:
    st.sidebar.info(
        "ℹ️ AgentOps is optional. Add AGENTOPS_API_KEY to enable it."
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Document Status")

if "db_client" not in st.session_state:
    (
        st.session_state.db_client,
        st.session_state.model,
        st.session_state.gemini_client,
    ) = initialize_rag_dependencies()

collection = get_collection()
knowledge_base_count = collection.count() if collection else 0

st.sidebar.info(f"Loaded Records: {knowledge_base_count}")
st.sidebar.info(
    f"Visual PDF Pages Indexed: "
    f"{st.session_state.visual_index_count}"
)
st.sidebar.caption(
    f"CAG Cache Size: {len(st.session_state.cache)}"
)

if st.sidebar.button(
    "Clear RAG Storage & Cache",
    use_container_width=True,
):
    clear_rag_storage()

st.sidebar.markdown("---")
st.sidebar.subheader("Response Options")

response_mode = st.sidebar.selectbox(
    "Response mode",
    ["Text", "Voice"],
)

tts_engine = st.sidebar.selectbox(
    "TTS engine",
    ["Edge-TTS", "gTTS"],
)

language_display = st.sidebar.selectbox(
    "Answer Language",
    list(LANGUAGE_DICT.keys()),
    index=0,
)

st.session_state.selected_language = language_display
language_code = LANGUAGE_DICT.get(
    language_display,
    "en",
)


# -----------------------------------------------------------
# Module 1: Document Loader
# -----------------------------------------------------------

if menu == "Document Loader":
    st.title("Document Loader 📄➡️🧠")

    st.markdown(
        "## Text RAG + ColPali-Inspired Visual Page Retrieval"
    )

    st.caption(
        "For PDFs, the app extracts text chunks and renders pages as images. "
        "Gemini generates a searchable description for each page. "
        "At question time, the relevant PDF page image is retrieved and "
        "sent directly to Gemini."
    )

    st.warning(
        "For diagrams, charts, tables, and visual architecture documents, "
        "upload the document as a PDF. URL ingestion currently indexes text "
        "only, because external webpages are not automatically screenshot."
    )

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Upload Files")

        max_pdf_pages = st.number_input(
            "Maximum PDF pages to process visually",
            min_value=1,
            max_value=25,
            value=DEFAULT_PDF_PAGES_TO_PROCESS,
            step=1,
            help=(
                "Each page uses a Gemini visual-analysis request during "
                "ingestion. Start with 5–10 pages to control API usage."
            ),
        )

        uploaded_files = st.file_uploader(
            "Upload Files (PDF, TXT, CSV, HTML, XML, JSON)",
            type=[
                "pdf",
                "txt",
                "csv",
                "html",
                "htm",
                "xml",
                "json",
            ],
            accept_multiple_files=True,
        )

        if uploaded_files:
            if st.button(
                f"Process {len(uploaded_files)} File(s) and Ingest",
                use_container_width=True,
            ):
                total_text_records = 0
                total_visual_records = 0

                with st.spinner(
                    "Extracting text, rendering PDF pages, and indexing visuals..."
                ):
                    for uploaded_file in uploaded_files:
                        file_bytes = uploaded_file.getvalue()
                        file_hash = hashlib.sha256(
                            file_bytes
                        ).hexdigest()

                        if file_hash in st.session_state.processed_file_hashes:
                            st.warning(
                                f"Skipped: '{uploaded_file.name}' "
                                "was already processed in this session."
                            )
                            continue

                        try:
                            if uploaded_file.name.lower().endswith(".pdf"):
                                (
                                    text_count,
                                    visual_count,
                                    _,
                                ) = process_pdf_for_visual_rag(
                                    uploaded_file=uploaded_file,
                                    max_pages_to_render=int(max_pdf_pages),
                                )

                                total_text_records += text_count
                                total_visual_records += visual_count

                                st.session_state.visual_index_count += (
                                    visual_count
                                )

                                st.success(
                                    f"Processed PDF: {uploaded_file.name} "
                                    f"— {text_count} text chunks and "
                                    f"{visual_count} visual page records."
                                )

                                tracker.track_document_upload(
                                    filename=uploaded_file.name,
                                    file_size=len(file_bytes),
                                    chunk_count=text_count + visual_count,
                                    extra={
                                        "text_chunks": text_count,
                                        "visual_pages": visual_count,
                                        "file_type": "pdf",
                                    },
                                )

                            else:
                                text_count, _ = process_text_document(
                                    uploaded_file=uploaded_file,
                                )

                                total_text_records += text_count

                                st.success(
                                    f"Processed: {uploaded_file.name} "
                                    f"— {text_count} text chunks."
                                )

                                tracker.track_document_upload(
                                    filename=uploaded_file.name,
                                    file_size=len(file_bytes),
                                    chunk_count=text_count,
                                    extra={
                                        "text_chunks": text_count,
                                        "visual_pages": 0,
                                        "file_type": "text",
                                    },
                                )

                            st.session_state.ingested_files.append(
                                uploaded_file.name
                            )

                            st.session_state.processed_file_hashes.add(
                                file_hash
                            )

                        except Exception as error:
                            st.error(
                                f"Failed to process "
                                f"'{uploaded_file.name}': {error}"
                            )

                            tracker.track_error(
                                operation="document_ingestion",
                                error_message=str(error),
                            )

                st.success(
                    f"Ingestion complete: {total_text_records} text records "
                    f"and {total_visual_records} visual page records."
                )

    with right_column:
        st.subheader("Load from URL 🌐")

        url_input = st.text_input(
            "Enter a website URL (http/https)",
            placeholder="https://example.com/article",
        )

        st.info(
            "URL ingestion currently performs text retrieval. "
            "For direct visual-page retrieval, download visual documents "
            "as PDF and upload them in the left panel."
        )

        if st.button(
            "Fetch URL and Ingest",
            use_container_width=True,
        ):
            if not url_input.strip():
                st.warning("Enter a valid URL first.")
            else:
                with st.spinner(
                    f"Fetching and indexing: {url_input.strip()}"
                ):
                    try:
                        added_records = process_url_document(
                            url=url_input.strip(),
                        )

                        st.session_state.ingested_files.append(
                            url_input.strip()
                        )

                        st.success(
                            f"Ingested {added_records} text chunks from URL."
                        )

                        tracker.track_document_upload(
                            filename=url_input.strip(),
                            file_size=0,
                            chunk_count=added_records,
                            extra={
                                "file_type": "url",
                                "visual_pages": 0,
                            },
                        )

                    except Exception as error:
                        st.error(f"URL ingestion failed: {error}")

                        tracker.track_error(
                            operation="url_ingestion",
                            error_message=str(error),
                        )

    st.markdown("---")
    st.subheader("Currently Ingested Files / URLs")

    if st.session_state.ingested_files:
        st.json(st.session_state.ingested_files)
    else:
        st.info(
            "No files or URLs have been loaded into the RAG system."
        )


# -----------------------------------------------------------
# Module 2: RAG Chatbot
# -----------------------------------------------------------

elif menu == "RAG Chatbot":
    st.title("RAG AI Agent 🧠")
    st.markdown(
        "### Text Retrieval + Visual PDF Page Retrieval"
    )

    st.caption(
        f"**Status:** Loaded Records: {knowledge_base_count} | "
        f"Visual Pages: {st.session_state.visual_index_count} | "
        f"Language: {st.session_state.selected_language} | "
        f"Mode: {response_mode} ({tts_engine})"
    )

    if not GEMINI_API_KEY:
        st.error(
            "Set GEMINI_API_KEY in Streamlit secrets before using the chatbot."
        )
        st.stop()

    retrieval_column, visual_column = st.columns(2)

    with retrieval_column:
        retrieval_count = st.slider(
            "Records to retrieve",
            min_value=1,
            max_value=12,
            value=DEFAULT_RETRIEVAL_COUNT,
        )

    with visual_column:
        maximum_visual_images = st.slider(
            "Maximum retrieved PDF page images for Gemini",
            min_value=0,
            max_value=6,
            value=DEFAULT_MAX_VISUAL_PAGES,
        )

    if not st.session_state.messages_rag:
        st.session_state.messages_rag = [
            {
                "role": "assistant",
                "content": (
                    "Hello! Upload documents in the Document Loader module. "
                    "For visual questions about diagrams, tables, or charts, "
                    "upload a PDF and ask your question here."
                ),
            }
        ]

    for message in st.session_state.messages_rag:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message.get("audio"):
                st.audio(
                    io.BytesIO(
                        base64.b64decode(message["audio"])
                    ),
                    format="audio/mp3",
                )

    user_question = st.chat_input(
        "Ask a question about your documents..."
    )

    if user_question:
        st.session_state.messages_rag.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            audio = None

            with st.spinner(
                "Retrieving text and relevant visual PDF pages..."
            ):
                start_time = time.time()

                (
                    answer,
                    retrieved_records,
                    retrieved_images,
                    served_from_cache,
                ) = rag_pipeline(
                    query=user_question,
                    selected_language=st.session_state.selected_language,
                    retrieval_count=int(retrieval_count),
                    maximum_visual_images=int(
                        maximum_visual_images
                    ),
                )

                response_time = time.time() - start_time

                if served_from_cache:
                    st.info(
                        "🔄 Serving response from cache "
                        "(CAG / cost reduction)."
                    )

                tracker.track_rag_query(
                    query=user_question,
                    retrieved_chunks=len(retrieved_records),
                    answer=answer,
                    response_time=response_time,
                    extra={
                        "model": GEMINI_MODEL,
                        "cache_hit": served_from_cache,
                    },
                )

                tracker.track_visual_rag_query(
                    query=user_question,
                    retrieved_chunks=len(retrieved_records),
                    retrieved_images=len(retrieved_images),
                    answer=answer,
                    response_time=response_time,
                    extra={
                        "model": GEMINI_MODEL,
                        "cache_hit": served_from_cache,
                    },
                )

                st.write(answer)

                st.caption(
                    f"Generation time: {response_time:.2f} seconds | "
                    f"Retrieved text/visual records: "
                    f"{len(retrieved_records)} | "
                    f"Retrieved PDF images: {len(retrieved_images)}"
                )

                if retrieved_images:
                    st.markdown(
                        "#### Retrieved Visual Pages Sent to Gemini"
                    )

                    image_columns = st.columns(
                        min(2, len(retrieved_images))
                    )

                    for image_index, image_data in enumerate(
                        retrieved_images
                    ):
                        with image_columns[
                            image_index % len(image_columns)
                        ]:
                            st.image(
                                image_data["image_bytes"],
                                caption=(
                                    f"{image_data['source_name']} "
                                    f"— PDF page "
                                    f"{image_data['page_number']}"
                                ),
                                use_container_width=True,
                            )

                if retrieved_records:
                    with st.expander(
                        "View retrieved text and visual descriptions"
                    ):
                        for record_index, record in enumerate(
                            retrieved_records,
                            start=1,
                        ):
                            metadata = record.get("metadata", {})

                            source_name = metadata.get(
                                "source_name",
                                "Unknown",
                            )

                            record_type = metadata.get(
                                "record_type",
                                "text",
                            )

                            page_number = metadata.get(
                                "page_number",
                                0,
                            )

                            st.markdown(
                                f"**{record_index}. {source_name}** "
                                f"| Type: `{record_type}` "
                                f"| Page: `{page_number}`"
                            )

                            st.caption(
                                record.get("document", "")[:1200]
                            )

                if response_mode == "Voice":
                    with st.spinner("Synthesizing speech..."):
                        audio, mime_type, tts_error = synthesize(
                            text=answer,
                            engine=tts_engine,
                            language_code=language_code,
                        )

                        if audio:
                            st.audio(
                                io.BytesIO(audio),
                                format=mime_type,
                            )

                            tracker.track_tts_generation(
                                text=answer,
                                language=st.session_state.selected_language,
                                engine=tts_engine,
                                audio_duration=0.0,
                            )
                        else:
                            st.warning(
                                f"TTS failed: {tts_error}"
                            )

            assistant_message = {
                "role": "assistant",
                "content": answer,
            }

            if audio:
                assistant_message["audio"] = base64.b64encode(
                    audio
                ).decode("utf-8")

            st.session_state.messages_rag.append(
                assistant_message
            )


# -----------------------------------------------------------
# Module 3: TTS Demo
# -----------------------------------------------------------

elif menu == "TTS Demo (Standalone)":
    st.title("Text-to-Speech Demo 🔊")

    st.info(
        "Uses the selected TTS engine and language from the sidebar."
    )

    tts_text = st.text_area(
        "Text to convert to speech",
        (
            "This is a demonstration of the multilingual "
            "text-to-speech capabilities of the RAG application."
        ),
        height=150,
    )

    if st.button(
        "Generate Speech",
        use_container_width=True,
    ):
        if not tts_text.strip():
            st.warning("Please enter text for TTS.")
        else:
            with st.spinner(
                f"Generating audio with {tts_engine}..."
            ):
                audio, mime_type, error = synthesize(
                    text=tts_text,
                    engine=tts_engine,
                    language_code=language_code,
                )

                if audio:
                    st.audio(
                        io.BytesIO(audio),
                        format=mime_type,
                    )

                    st.success("Speech generated successfully.")

                    tracker.track_tts_generation(
                        text=tts_text,
                        language=st.session_state.selected_language,
                        engine=tts_engine,
                        audio_duration=0.0,
                    )
                else:
                    st.error(
                        f"TTS generation failed: {error}"
                    )
