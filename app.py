import os
import tempfile
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOllama
from langchain.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
import streamlit as st
from langchain_community.tools.e2b_data_analysis.tool import UploadedFile
from streamlit_chat import message

models = {
    'mistral': {
        'model_name': 'mistral',
        # https://smith.langchain.com/hub/rlm/rag-prompt-mistral
        'prompt': """<s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. [/INST] </s> 
[INST] Question: {question} 
Context: {context} 
Answer: [/INST]"""
    },
    'llama2': {
        'model_name': 'llama2',
        # https://smith.langchain.com/hub/rlm/rag-prompt-llama
        'prompt': """[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> 
Question: {question} 
Context: {context} 
Answer: [/INST]"""
    }
}

model = 'mistral'


class ChatAssistant:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model=models[model]['model_name'])  # mistral
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(models[model]['prompt'])

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Add a PDF document first."
        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None


class Page:

    def __init__(self):
        st.set_page_config(page_title="Chatbot with PDF")
        if len(st.session_state) == 0:
            st.session_state["messages"] = []
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


if __name__ == "__main__":
    Page()
