# app.py
# 📚 ColPali-Inspired Visual RAG AI Agent with Multilingual & Voice Support 🎙️
# Modules preserved:
# 1. Document Loader
# 2. RAG Chatbot
# 3. TTS Demo (Standalone)
#
# Features:
# - Text RAG with ChromaDB + SentenceTransformer
# - PDF page rendering with PyMuPDF
# - Gemini visual page analysis for visual retrieval
# - Relevant PDF page images sent to Gemini
# - Persistent visual chat history after switching Streamlit modules
# - CAG response cache
# - Multilingual answer support
# - Edge-TTS and gTTS voice support
# - Optional AgentOps tracking

import asyncio
import base64
import hashlib
import io
import os
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Tuple

import chromadb
import fitz
import pandas as pd
import requests
import streamlit as st
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
# Optional SentenceTransformer import
# -----------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# -----------------------------------------------------------
# Gemini SDK import
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
# Optional TTS imports
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
# Streamlit configuration
# -----------------------------------------------------------

st.set_page_config(
    page_title="ColPali Visual RAG AI Agent",
    page_icon="📚",
    layout="wide",
)


# -----------------------------------------------------------
# App constants
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
# Cached dependencies
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
# ChromaDB helpers
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
            metadata={
                "description": "Text and visual PDF-page retrieval for RAG"
            },
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
            "Check requirements.txt for sentence-transformers."
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
    n_results: int,
) -> List[Dict[str, Any]]:
    collection = get_collection()

    if collection is None or collection.count() == 0:
        return []

    query_embedding = create_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    records = []

    for index, document in enumerate(documents):
        records.append(
            {
                "document": document,
                "metadata": (
                    metadatas[index]
                    if index < len(metadatas)
                    else {}
                ),
                "distance": (
                    distances[index]
                    if index < len(distances)
                    else None
                ),
            }
        )

    return records


# -----------------------------------------------------------
# Document and PDF helpers
# -----------------------------------------------------------

def create_document_id(file_name: str, file_bytes: bytes) -> str:
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:20]

    safe_name = "".join(
        character if character.isalnum() else "_"
        for character in file_name
    )

    return f"{safe_name}_{file_hash}"


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text_parts = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
        for page_number, page in enumerate(pdf_document, start=1):
            page_text = page.get_text("text").strip()

            if page_text:
                text_parts.append(
                    f"[PDF Page {page_number}]\n{page_text}"
                )

    return "\n\n".join(text_parts)


def render_pdf_pages(
    pdf_bytes: bytes,
    max_pages: int,
    zoom: float = 1.7,
) -> List[Tuple[int, bytes]]:
    output = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
        page_limit = min(pdf_document.page_count, max_pages)

        for page_index in range(page_limit):
            page = pdf_document.load_page(page_index)

            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom),
                alpha=False,
            )

            output.append(
                (
                    page_index + 1,
                    pixmap.tobytes("png"),
                )
            )

    return output


def extract_text_from_upload(uploaded_file) -> Tuple[str, bool]:
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


def load_text_from_url(url: str) -> Tuple[str, bool]:
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(compatible; ColPaliVisualRAG/1.0)"
                )
            },
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
    gemini_client = initialize_gemini_client()

    if gemini_client is None or types is None:
        return (
            f"PDF page {page_number} from {file_name}. "
            "Visual analysis unavailable because Gemini is not configured."
        )

    prompt = f"""
Analyze this PDF page for visual document retrieval.

Document: {file_name}
PDF page: {page_number}

Create a concise, searchable description of:
- headings and visible text
- tables, columns, rows, and numerical values
- charts, axes, trends, and labels
- diagrams, arrows, hierarchy, and component relationships
- icons, callouts, and visual structure

Do not invent content. Return only a useful retrieval description.
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
            f"Visual analysis fallback for page {page_number}: "
            f"{error}"
        )

    return f"PDF page {page_number} from {file_name}."


# -----------------------------------------------------------
# Ingestion functions
# -----------------------------------------------------------

def process_pdf_for_visual_rag(
    uploaded_file,
    max_pages_to_render: int,
) -> Tuple[int, int]:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    document_id = create_document_id(file_name, file_bytes)

    pdf_text = extract_text_from_pdf(file_bytes)
    text_chunks = split_documents(pdf_text)

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

    rendered_pages = render_pdf_pages(
        pdf_bytes=file_bytes,
        max_pages=max_pages_to_render,
    )

    if document_id not in st.session_state.page_images:
        st.session_state.page_images[document_id] = {}

    visual_documents = []
    visual_metadatas = []

    for page_number, image_bytes in rendered_pages:
        st.session_state.page_images[document_id][page_number] = image_bytes

        visual_description = analyze_pdf_page_with_gemini(
            image_bytes=image_bytes,
            page_number=page_number,
            file_name=file_name,
        )

        visual_documents.append(
            f"VISUAL PAGE DESCRIPTION\n"
            f"File: {file_name}\n"
            f"PDF Page: {page_number}\n\n"
            f"{visual_description}"
        )

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

    return text_count, visual_count


def process_text_document(uploaded_file) -> int:
    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    document_id = create_document_id(file_name, file_bytes)

    raw_text, success = extract_text_from_upload(uploaded_file)

    if not success:
        raise RuntimeError(raw_text)

    chunks = split_documents(raw_text)

    metadatas = [
        {
            "source_name": file_name,
            "document_id": document_id,
            "record_type": "text",
            "page_number": 0,
        }
        for _ in chunks
    ]

    return store_records(
        documents=chunks,
        metadatas=metadatas,
    )


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
# Retrieval and visual-page lookup
# -----------------------------------------------------------

def get_retrieved_page_images(
    retrieved_records: List[Dict[str, Any]],
    maximum_images: int,
) -> List[Dict[str, Any]]:
    selected_images = []
    selected_keys = set()

    for record in retrieved_records:
        metadata = record.get("metadata", {})

        if metadata.get("record_type") != "visual_page":
            continue

        document_id = metadata.get("document_id")
        page_number = metadata.get("page_number")
        source_name = metadata.get("source_name", "Unknown file")

        key = (document_id, page_number)

        if key in selected_keys:
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
                    "source_name": source_name,
                    "page_number": page_number,
                    "document_id": document_id,
                }
            )

            selected_keys.add(key)

        if len(selected_images) >= maximum_images:
            break

    return selected_images


def build_text_context(
    retrieved_records: List[Dict[str, Any]],
) -> str:
    if not retrieved_records:
        return "No relevant content was retrieved."

    context_parts = []

    for index, record in enumerate(retrieved_records, start=1):
        metadata = record.get("metadata", {})

        source_name = metadata.get("source_name", "Unknown")
        record_type = metadata.get("record_type", "text")
        page_number = metadata.get("page_number", 0)

        context_parts.append(
            f"[Source {index} | File: {source_name} | "
            f"Type: {record_type} | PDF page: {page_number}]\n"
            f"{record.get('document', '')}"
        )

    return "\n\n---\n\n".join(context_parts)


# -----------------------------------------------------------
# Gemini visual RAG answer generation
# -----------------------------------------------------------

def call_gemini_visual_rag(
    query: str,
    text_context: str,
    retrieved_images: List[Dict[str, Any]],
    selected_language: str,
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
You are an accurate multimodal RAG assistant.

Use only the retrieved text context and any supplied PDF page images.

Rules:
- Analyze diagrams, charts, tables, labels, and visual relationships carefully.
- Do not invent facts not present in the supplied sources.
- Mention the source file and PDF page number whenever possible.
- If the evidence is insufficient, state that clearly.
- Reply in {selected_language}.
""".strip()

    prompt = f"""
RETRIEVED CONTEXT:
{text_context}

USER QUESTION:
{query}
""".strip()

    contents: List[Any] = [prompt]

    for image_data in retrieved_images:
        contents.append(
            f"Retrieved visual source: "
            f"{image_data['source_name']}, "
            f"PDF page {image_data['page_number']}."
        )

        contents.append(
            types.Part.from_bytes(
                data=image_data["image_bytes"],
                mime_type="image/png",
            )
        )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                top_p=0.8,
                max_output_tokens=2048,
            ),
        )

        if response.text:
            return {"response": response.text.strip()}

        return {"error": "Gemini returned an empty response."}

    except APIError as error:
        return {"error": str(error)}

    except Exception as error:
        return {"error": str(error)}


# -----------------------------------------------------------
# CAG cache and RAG pipeline
# -----------------------------------------------------------

def create_cache_key(
    query: str,
    selected_language: str,
    retrieval_count: int,
    maximum_visual_images: int,
) -> str:
    raw_key = (
        f"{query}|{selected_language}|{retrieval_count}|"
        f"{maximum_visual_images}|{GEMINI_MODEL}"
    )

    return hashlib.sha256(
        raw_key.encode("utf-8")
    ).hexdigest()


def rag_pipeline(
    query: str,
    selected_language: str,
    retrieval_count: int,
    maximum_visual_images: int,
) -> Tuple[
    str,
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    bool,
]:
    cache_key = create_cache_key(
        query=query,
        selected_language=selected_language,
        retrieval_count=retrieval_count,
        maximum_visual_images=maximum_visual_images,
    )

    cache = st.session_state.cache

    if (
        cache_key in cache
        and time.time() - cache[cache_key]["timestamp"]
        < CACHE_EXPIRY_SECONDS
    ):
        cached_result = cache[cache_key]

        return (
            cached_result["answer"],
            cached_result["retrieved_records"],
            cached_result["retrieved_images"],
            True,
        )

    retrieved_records = retrieve_records(
        query=query,
        n_results=retrieval_count,
    )

    retrieved_images = get_retrieved_page_images(
        retrieved_records=retrieved_records,
        maximum_images=maximum_visual_images,
    )

    text_context = build_text_context(
        retrieved_records=retrieved_records,
    )

    response = call_gemini_visual_rag(
        query=query,
        text_context=text_context,
        retrieved_images=retrieved_images,
        selected_language=selected_language,
    )

    answer = response.get(
        "response",
        f"Generation error: {response.get('error', 'Unknown error')}",
    )

    st.session_state.cache[cache_key] = {
        "answer": answer,
        "timestamp": time.time(),
        "retrieved_records": retrieved_records,
        "retrieved_images": retrieved_images,
    }

    return (
        answer,
        retrieved_records,
        retrieved_images,
        False,
    )


# -----------------------------------------------------------
# TTS functions
# -----------------------------------------------------------

async def edge_tts_async(
    text: str,
    voice: str,
):
    if edge_tts is None:
        return None

    communication = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="+0%",
    )

    output = io.BytesIO()

    async for chunk in communication.stream():
        if chunk["type"] == "audio":
            output.write(chunk["data"])

    return output.getvalue()


def tts_edge(text: str, voice: str):
    try:
        return asyncio.run(
            edge_tts_async(text=text, voice=voice)
        ), None
    except Exception as error:
        return None, str(error)


def tts_gtts(text: str, language_code: str):
    if gTTS is None:
        return None, "gTTS is not available."

    try:
        buffer = io.BytesIO()

        gTTS(
            text=text,
            lang=language_code,
        ).write_to_fp(buffer)

        return buffer.getvalue(), None

    except Exception as error:
        return None, str(error)


def synthesize(
    text: str,
    engine: str,
    language_code: str,
):
    voice_map = {
        "en": "en-US-AriaNeural",
        "hi": "hi-IN-SwaraNeural",
        "ta": "ta-IN-PallaviNeural",
        "bn": "bn-IN-TanishaaNeural",
        "es": "es-ES-ElviraNeural",
        "fr": "fr-FR-DeniseNeural",
        "de": "de-DE-KatjaNeural",
        "ar": "ar-SA-ZariyahNeural",
        "zh-Hans": "zh-CN-XiaoxiaoNeural",
        "ja": "ja-JP-NanamiNeural",
        "ko": "ko-KR-SunHiNeural",
        "pt": "pt-PT-RaquelNeural",
        "it": "it-IT-ElsaNeural",
        "nl": "nl-NL-ColetteNeural",
        "tr": "tr-TR-EmelNeural",
        "ru": "ru-RU-SvetlanaNeural",
    }

    if engine == "Edge-TTS":
        voice = voice_map.get(
            language_code,
            "en-US-AriaNeural",
        )

        audio, error = tts_edge(
            text=text,
            voice=voice,
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
# Persistent visual chat rendering
# -----------------------------------------------------------

def render_visual_pages(
    retrieved_images: List[Dict[str, Any]],
) -> None:
    if not retrieved_images:
        return

    st.markdown("#### Retrieved Visual Pages Sent to Gemini")

    image_columns = st.columns(
        min(2, len(retrieved_images))
    )

    for image_index, image_data in enumerate(retrieved_images):
        with image_columns[image_index % len(image_columns)]:
            st.image(
                image_data["image_bytes"],
                caption=(
                    f"{image_data['source_name']} — "
                    f"PDF page {image_data['page_number']}"
                ),
                use_container_width=True,
            )


def render_retrieval_details(
    retrieved_records: List[Dict[str, Any]],
) -> None:
    if not retrieved_records:
        return

    with st.expander("View retrieved text and visual descriptions"):
        for index, record in enumerate(
            retrieved_records,
            start=1,
        ):
            metadata = record.get("metadata", {})

            st.markdown(
                f"**{index}. {metadata.get('source_name', 'Unknown')}** "
                f"| Type: `{metadata.get('record_type', 'text')}` "
                f"| Page: `{metadata.get('page_number', 0)}`"
            )

            st.caption(
                record.get("document", "")[:1200]
            )


def render_chat_history() -> None:
    """
    Renders all past messages including persisted PDF images.

    This runs every time the RAG Chatbot module is opened, so answers
    and visual pages remain visible after switching modules.
    """

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

            if message["role"] == "assistant":
                render_visual_pages(
                    message.get("retrieved_images", [])
                )

                render_retrieval_details(
                    message.get("retrieved_records", [])
                )

                if message.get("response_time") is not None:
                    st.caption(
                        f"Generation time: "
                        f"{message['response_time']:.2f} seconds | "
                        f"Retrieved records: "
                        f"{len(message.get('retrieved_records', []))} | "
                        f"Retrieved PDF images: "
                        f"{len(message.get('retrieved_images', []))}"
                    )


# -----------------------------------------------------------
# Storage cleanup
# -----------------------------------------------------------

def clear_chat_and_visual_history() -> None:
    st.session_state.messages_rag = []
    st.session_state.cache = {}
    st.rerun()


def clear_rag_storage() -> None:
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
                name=COLLECTION_NAME
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
    session_name = (
        f"{tracker.session_id[:8]}..."
        if tracker.session_id
        else "active"
    )

    st.sidebar.success(
        f"✅ AgentOps session: {session_name}"
    )

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
loaded_record_count = collection.count() if collection else 0

st.sidebar.info(f"Loaded Records: {loaded_record_count}")
st.sidebar.info(
    f"Visual PDF Pages Indexed: "
    f"{st.session_state.visual_index_count}"
)
st.sidebar.caption(
    f"CAG Cache Size: {len(st.session_state.cache)}"
)

if st.sidebar.button(
    "Clear Chat & Visual History",
    use_container_width=True,
):
    clear_chat_and_visual_history()

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
        "PDF uploads are indexed as both text chunks and visual page "
        "descriptions. Relevant page images are retrieved and sent to Gemini "
        "when you ask visual questions."
    )

    st.warning(
        "For tables, charts, diagrams, and image-heavy documentation, "
        "upload a PDF. URL ingestion currently indexes text only."
    )

    upload_column, url_column = st.columns(2)

    with upload_column:
        st.subheader("Upload Files")

        max_pdf_pages = st.number_input(
            "Maximum PDF pages to process visually",
            min_value=1,
            max_value=25,
            value=DEFAULT_PDF_PAGES_TO_PROCESS,
            step=1,
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
                    "Extracting text, rendering PDF images, and indexing..."
                ):
                    for uploaded_file in uploaded_files:
                        file_bytes = uploaded_file.getvalue()
                        file_hash = hashlib.sha256(
                            file_bytes
                        ).hexdigest()

                        if (
                            file_hash
                            in st.session_state.processed_file_hashes
                        ):
                            st.warning(
                                f"Skipped '{uploaded_file.name}': "
                                "already processed in this session."
                            )
                            continue

                        try:
                            if uploaded_file.name.lower().endswith(".pdf"):
                                (
                                    text_count,
                                    visual_count,
                                ) = process_pdf_for_visual_rag(
                                    uploaded_file=uploaded_file,
                                    max_pages_to_render=int(
                                        max_pdf_pages
                                    ),
                                )

                                total_text_records += text_count
                                total_visual_records += visual_count

                                st.session_state.visual_index_count += (
                                    visual_count
                                )

                                st.success(
                                    f"Processed PDF: {uploaded_file.name} "
                                    f"— {text_count} text chunks and "
                                    f"{visual_count} visual pages."
                                )

                                tracker.track_document_upload(
                                    filename=uploaded_file.name,
                                    file_size=len(file_bytes),
                                    chunk_count=(
                                        text_count + visual_count
                                    ),
                                    extra={
                                        "file_type": "pdf",
                                        "text_chunks": text_count,
                                        "visual_pages": visual_count,
                                    },
                                )

                            else:
                                text_count = process_text_document(
                                    uploaded_file=uploaded_file
                                )

                                total_text_records += text_count

                                st.success(
                                    f"Processed {uploaded_file.name} "
                                    f"— {text_count} text chunks."
                                )

                                tracker.track_document_upload(
                                    filename=uploaded_file.name,
                                    file_size=len(file_bytes),
                                    chunk_count=text_count,
                                    extra={
                                        "file_type": "text",
                                        "visual_pages": 0,
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
                                f"Failed to process {uploaded_file.name}: "
                                f"{error}"
                            )

                            tracker.track_error(
                                operation="document_ingestion",
                                error_message=str(error),
                            )

                st.success(
                    f"Ingestion completed: {total_text_records} text "
                    f"records and {total_visual_records} visual records."
                )

    with url_column:
        st.subheader("Load from URL 🌐")

        url_input = st.text_input(
            "Enter a website URL (http/https)",
            placeholder="https://example.com/article",
        )

        st.info(
            "URLs are indexed as text. For direct image retrieval, "
            "download visual documents as PDF and upload them."
        )

        if st.button(
            "Fetch URL and Ingest",
            use_container_width=True,
        ):
            if not url_input.strip():
                st.warning("Enter a valid URL first.")
            else:
                with st.spinner("Fetching and indexing URL..."):
                    try:
                        count = process_url_document(
                            url_input.strip()
                        )

                        st.session_state.ingested_files.append(
                            url_input.strip()
                        )

                        st.success(
                            f"Ingested {count} text chunks from the URL."
                        )

                        tracker.track_document_upload(
                            filename=url_input.strip(),
                            file_size=0,
                            chunk_count=count,
                            extra={
                                "file_type": "url",
                                "visual_pages": 0,
                            },
                        )

                    except Exception as error:
                        st.error(
                            f"URL ingestion failed: {error}"
                        )

                        tracker.track_error(
                            operation="url_ingestion",
                            error_message=str(error),
                        )

    st.markdown("---")
    st.subheader("Currently Ingested Files / URLs")

    if st.session_state.ingested_files:
        st.json(st.session_state.ingested_files)
    else:
        st.info("No files or URLs have been loaded yet.")


# -----------------------------------------------------------
# Module 2: RAG Chatbot
# -----------------------------------------------------------

elif menu == "RAG Chatbot":
    st.title("RAG AI Agent 🧠")
    st.markdown(
        "### Text Retrieval + Persistent Visual PDF Page Retrieval"
    )

    st.caption(
        f"Loaded Records: {loaded_record_count} | "
        f"Visual Pages: {st.session_state.visual_index_count} | "
        f"Language: {st.session_state.selected_language} | "
        f"Mode: {response_mode} ({tts_engine})"
    )

    if not GEMINI_API_KEY:
        st.error(
            "Set GEMINI_API_KEY in Streamlit secrets to use the chatbot."
        )
        st.stop()

    setting_column_1, setting_column_2 = st.columns(2)

    with setting_column_1:
        retrieval_count = st.slider(
            "Records to retrieve",
            min_value=1,
            max_value=12,
            value=DEFAULT_RETRIEVAL_COUNT,
        )

    with setting_column_2:
        maximum_visual_images = st.slider(
            "Maximum retrieved PDF images for Gemini",
            min_value=0,
            max_value=6,
            value=DEFAULT_MAX_VISUAL_PAGES,
        )

    if not st.session_state.messages_rag:
        st.session_state.messages_rag = [
            {
                "role": "assistant",
                "content": (
                    "Hello! Upload a document through Document Loader. "
                    "For diagrams, tables, charts, and other visuals, upload "
                    "a PDF and ask a question here."
                ),
            }
        ]

    # This renders text, retrieved source details, and actual page images
    # every time the user returns to RAG Chatbot.
    render_chat_history()

    user_question = st.chat_input(
        "Ask a question about your documents..."
    )

    if user_question:
        user_message = {
            "role": "user",
            "content": user_question,
        }

        st.session_state.messages_rag.append(user_message)

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            audio = None

            with st.spinner(
                "Retrieving text and relevant PDF page visuals..."
            ):
                started_at = time.time()

                (
                    answer,
                    retrieved_records,
                    retrieved_images,
                    cache_hit,
                ) = rag_pipeline(
                    query=user_question,
                    selected_language=st.session_state.selected_language,
                    retrieval_count=int(retrieval_count),
                    maximum_visual_images=int(
                        maximum_visual_images
                    ),
                )

                response_time = time.time() - started_at

                if cache_hit:
                    st.info(
                        "🔄 Serving response from CAG cache."
                    )

                tracker.track_rag_query(
                    query=user_question,
                    retrieved_chunks=len(retrieved_records),
                    answer=answer,
                    response_time=response_time,
                    extra={
                        "model": GEMINI_MODEL,
                        "cache_hit": cache_hit,
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
                        "cache_hit": cache_hit,
                    },
                )

                st.write(answer)

                render_visual_pages(retrieved_images)

                render_retrieval_details(retrieved_records)

                st.caption(
                    f"Generation time: {response_time:.2f} seconds | "
                    f"Retrieved records: {len(retrieved_records)} | "
                    f"Retrieved PDF images: {len(retrieved_images)}"
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
                                language=(
                                    st.session_state.selected_language
                                ),
                                engine=tts_engine,
                                audio_duration=0.0,
                            )
                        else:
                            st.warning(
                                f"TTS failed: {tts_error}"
                            )

            # Critical fix:
            # Save answer AND its retrieved image bytes + retrieval details.
            # Therefore the visuals still appear after changing modules.
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "retrieved_images": retrieved_images,
                "retrieved_records": retrieved_records,
                "response_time": response_time,
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
        "Uses the current TTS engine and language selected in the sidebar."
    )

    tts_text = st.text_area(
        "Text to convert to speech",
        (
            "This is a demonstration of the multilingual "
            "text-to-speech capabilities of the RAG AI Agent."
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
                        language=(
                            st.session_state.selected_language
                        ),
                        engine=tts_engine,
                        audio_duration=0.0,
                    )
                else:
                    st.error(
                        f"TTS generation failed: {error}"
                    )
