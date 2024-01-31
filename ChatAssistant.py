from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOllama
from langchain.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

import KnowledgeManager


class ChatAssistant:

    def __init__(self, knowledge_manager: KnowledgeManager):
        self.knowledge_manager = knowledge_manager

    def ask(self, query: str):
        return self.knowledge_manager.query(query)

    def clear(self):
        pass
