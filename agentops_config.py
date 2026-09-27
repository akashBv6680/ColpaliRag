import time
from typing import Any, Dict, List, Optional

import streamlit as st

try:
    import agentops
    AGENTOPS_AVAILABLE = True
except Exception:
    AGENTOPS_AVAILABLE = False


class RAGAgentOpsTracker:
    """
    Tracker for RAG operations using AgentOps.
    Supports:
      - document_upload
      - rag_query
      - visual_rag_query (for ColPali-style calls with images)
      - tts_generation
      - error events
    """

    def __init__(self):
        self.initialized = False
        self.session_id: Optional[str] = None
        self.operations: List[Dict[str, Any]] = []

        if not AGENTOPS_AVAILABLE:
            st.warning("AgentOps package not available; monitoring will be disabled.")
            return

        api_key = st.secrets.get("AGENTOPS_API_KEY", None)
        if not api_key:
            st.warning("AGENTOPS_API_KEY not found in secrets; AgentOps will not record events.")
            return

        try:
            agentops.init(
                api_key=api_key,
                tags=["colpali-rag", "gemini", "visual-rag"],
            )
            self.initialized = True
            # Store session id if available
            self.session_id = getattr(agentops, "session_id", None)
        except Exception as e:
            st.warning(f"Failed to initialize AgentOps: {e}")
            self.initialized = False

    def _record(self, event_name: str, data: Dict[str, Any]):
        if not self.initialized:
            return
        try:
            agentops.record(event_name, data)
            self.operations.append({"event": event_name, "data": data, "timestamp": time.time()})
        except Exception as e:
            st.warning(f"Failed to record AgentOps event '{event_name}': {e}")

    def track_document_upload(
        self,
        filename: str,
        size: int,
        chunk_count: int,
        extra: Optional[Dict[str, Any]] = None
    ):
        data = {
            "filename": filename,
            "size_bytes": size,
            "chunk_count": chunk_count,
        }
        if extra:
            data.update(extra)
        self._record("document_upload", data)

    def track_rag_query(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        answer_length: int,
        response_time: float,
        extra: Optional[Dict[str, Any]] = None
    ):
        data = {
            "query": query,
            "num_chunks": len(retrieved_chunks),
            "answer_length": answer_length,
            "response_time_sec": response_time,
        }
        if extra:
            data.update(extra)
        self._record("rag_query", data)

    def track_visual_rag_query(
        self,
        query: str,
        num_chunks: int,
        num_images: int,
        answer_length: int,
        response_time: float,
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Specialized tracking for ColPali-style visual RAG calls
        where page images are sent to Gemini along with text context.
        """
        data = {
            "query": query,
            "num_chunks": num_chunks,
            "num_images": num_images,
            "answer_length": answer_length,
            "response_time_sec": response_time,
        }
        if extra:
            data.update(extra)
        self._record("visual_rag_query", data)

    def track_tts_generation(
        self,
        text_length: int,
        language: str,
        engine: str,
        audio_duration: float,
        extra: Optional[Dict[str, Any]] = None
    ):
        data = {
            "text_length": text_length,
            "language": language,
            "engine": engine,
            "audio_duration_sec": audio_duration,
        }
        if extra:
            data.update(extra)
        self._record("tts_generation", data)

    def track_error(
        self,
        operation: str,
        error_message: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        data = {
            "operation": operation,
            "error_message": error_message,
        }
        if extra:
            data.update(extra)
        self._record("error", data)

    def get_session_dashboard_url(self) -> Optional[str]:
        if not self.initialized or not self.session_id:
            return None
        return f"https://app.agentops.ai/sessions/{self.session_id}"

    def end_session(self):
        if not self.initialized:
            return
        try:
            agentops.end_session()
        except Exception:
            pass


# Global tracker instance
tracker = RAGAgentOpsTracker()
