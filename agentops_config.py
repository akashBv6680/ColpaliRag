import streamlit as st
from datetime import datetime


class RAGAgentOpsTracker:
    """
    Safe operation tracker for the RAG application.

    Important:
    - Keeps local operation data in self.operations.
    - Initializes AgentOps when an API key is available.
    - Does not call agentops.record(), because the installed AgentOps
      version has a different record() signature and caused:
      'record() takes 1 positional argument but 2 were given'.
    """

    def __init__(self):
        self.session_id = None
        self.is_initialized = False
        self.operations = []
        self.agentops = None

    def initialize(self):
        if self.is_initialized:
            return True

        try:
            import agentops

            api_key = st.secrets.get("AGENTOPS_API_KEY")

            if not api_key:
                return False

            session = agentops.init(api_key=api_key)

            self.agentops = agentops
            self.is_initialized = True

            if hasattr(session, "session_id"):
                self.session_id = str(session.session_id)
            elif hasattr(agentops, "session_id"):
                self.session_id = str(agentops.session_id)
            else:
                self.session_id = None

            return True

        except Exception as error:
            print(f"AgentOps initialization warning: {error}")
            self.is_initialized = False
            return False

    def _save_operation(self, operation_name, data):
        operation = {
            "operation": operation_name,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            **data,
        }

        self.operations.append(operation)

    def track_document_upload(self, filename, file_size, chunk_count, extra=None):
        data = {
            "file_name": filename,
            "file_size_bytes": file_size,
            "chunks_created": chunk_count,
        }

        if extra:
            data.update(extra)

        self._save_operation("document_upload", data)

    def track_rag_query(self, query, retrieved_chunks, answer, response_time, extra=None):
        data = {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "response_time_seconds": round(response_time, 2),
            "answer_length": len(answer),
        }

        if extra:
            data.update(extra)

        self._save_operation("rag_query", data)

    def track_visual_rag_query(
        self,
        query,
        retrieved_chunks,
        retrieved_images,
        answer,
        response_time,
        extra=None,
    ):
        data = {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_images": retrieved_images,
            "response_time_seconds": round(response_time, 2),
            "answer_length": len(answer),
        }

        if extra:
            data.update(extra)

        self._save_operation("visual_rag_query", data)

    def track_tts_generation(self, text, language, engine, audio_duration):
        self._save_operation(
            "tts_generation",
            {
                "text_length": len(text),
                "language": language,
                "tts_engine": engine,
                "audio_duration_seconds": audio_duration,
            },
        )

    def track_error(self, operation, error_message):
        self.operations.append(
            {
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "error": error_message,
                "status": "failed",
            }
        )

    def get_session_dashboard_url(self):
        if self.session_id:
            return f"https://app.agentops.ai/sessions/{self.session_id}"

        return None

    def end_session(self):
        try:
            if self.is_initialized and self.agentops:
                self.agentops.end_session()
        except Exception as error:
            print(f"AgentOps end-session warning: {error}")


tracker = RAGAgentOpsTracker()
