import os
import tempfile
import streamlit as st
from langchain_community.tools.e2b_data_analysis.tool import UploadedFile
from streamlit_chat import message

class Page:

    def __init__(self):
        st.set_page_config(page_title="Chatbot with PDF")
        if len(st.session_state) == 0:
            st.session_state["messages"] = []
            from ChatAssistant import ChatAssistant
            st.session_state["assistant"] = ChatAssistant()

        st.header("Chatbot with PDF")
        st.text("The the PDFs that will be used to create the context")
        st.file_uploader(
            "Upload document",
            type=["pdf"],
            key="file_uploader",
            on_change=self._read_and_save_files,
            label_visibility="collapsed",
            accept_multiple_files=True
        )

        st.session_state["ingestion_spinner"] = st.empty()

        st.subheader("Chat")
        for i, (msg, is_user) in enumerate(st.session_state["messages"]):
            message(msg, is_user=is_user, key=str(i))
        st.session_state["thinking_spinner"] = st.empty()

        st.text_input("Message", key="user_input", on_change=self._process_input)

    def _process_input(self):
        if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
            self._process_user_text(st.session_state["user_input"].strip())

    def _process_user_text(self, user_text: str):
        with st.session_state["thinking_spinner"], st.spinner("Thinking"):
            assistant_reply = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((assistant_reply, False))

    def _read_and_save_files(self):
        self._clear_state()

        for file in st.session_state["file_uploader"]:
            self._read_and_save_file(file)

    def _read_and_save_file(self, file: UploadedFile):
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)

    def _clear_state(self):
        st.session_state["assistant"].clear()
        st.session_state["messages"] = []
        st.session_state["user_input"] = ""