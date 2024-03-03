import re
from abc import ABC

from chromadb import ClientAPI
from llama_index.legacy import VectorStoreIndex, ServiceContext, ChatPromptTemplate
from llama_index.legacy.core.base_retriever import BaseRetriever
from llama_index.legacy.llms import ChatMessage, MessageRole
from llama_index.legacy.query_engine import RetrieverQueryEngine
from llama_index.legacy.response_synthesizers import ResponseMode
from llama_index.legacy.vector_stores import ChromaVectorStore

from db import DB
from execution_context import ExecutionContext
from nullRetriever import NullRetriever
from prompts import Prompts, SummaryPrompts, DiagnosisPrompts, ChatPrompts


class Step(ABC):
    def __init__(self,
                 prompts: Prompts,
                 db: DB,
                 execution_context: ExecutionContext):
        self._prompts = prompts
        self._db = db
        self._execution_context = execution_context
        self._query_engine = None

    def query(self, query: str):
        if self._query_engine is None:
            self._query_engine = self._get_query_engine()
        response = self._query_engine.query(query)
        self._execution_context.handle(response)
        return response.response

    def _get_query_engine(self) -> RetrieverQueryEngine:
        vector_retriever_chunk = self._get_retriever()
        text_template = self._get_prompt_template(system_prompt=self._prompts.system_prompt,
                                                  user_prompt=self._prompts.user_prompt)
        return RetrieverQueryEngine.from_args(
            vector_retriever_chunk,
            service_context=self._execution_context.get_service_context(),
            verbose=True,
            response_mode=ResponseMode.COMPACT,
            text_qa_template=text_template)

    @staticmethod
    def _get_retriever() -> BaseRetriever:
        return NullRetriever()

    @staticmethod
    def _get_prompt_template(system_prompt: str, user_prompt: str):
        chat_text_qa_msgs = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=user_prompt,
            ),
        ]
        return ChatPromptTemplate(chat_text_qa_msgs)


class ChatGenerationStep(Step):
    def __init__(self, db: DB, execution_context: ExecutionContext):
        super().__init__(prompts=ChatPrompts(), db=db, execution_context=execution_context)

    def query(self, query: str):
        if not query:
            return 'How old are you?'
        next_question = super().query(query)
        return self._clean_question(next_question)

    @staticmethod
    def _clean_question(next_question):
        tmp = next_question.split("\n")
        next_question_clean = tmp[len(tmp) - 1]
        next_question_clean = next_question_clean.replace('You: ', '')
        next_question_clean = re.search('"?([^"]*)"?', next_question_clean).group(1)
        next_question_clean = next_question_clean.replace('"', '')
        next_question_clean = next_question_clean.strip()
        print(f'DEBUG:\n  next_question:{next_question}\n  next_question_clean:{next_question_clean}')
        return next_question_clean if next_question_clean else next_question


class SummaryGenerationStep(Step):
    def __init__(self, db: DB, execution_context: ExecutionContext):
        super().__init__(prompts=SummaryPrompts(), db=db, execution_context=execution_context)


class DiagnosisGenerationStep(Step):
    def __init__(self, db: DB, execution_context: ExecutionContext):
        super().__init__(prompts=DiagnosisPrompts(), db=db, execution_context=execution_context)

    def _get_vector_retriever_chunk(self, collection: str, db: ClientAPI,
                                    service_context: ServiceContext) -> BaseRetriever:
        chroma_collection = db.get_or_create_collection(collection)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        vector_store_index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
        return vector_store_index.as_retriever(similarity_top_k=self._db.retrieve_n_chunks)
