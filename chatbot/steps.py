"""
LLM RAG Chatbot
"""
from abc import ABC

import httpx
from chromadb import ClientAPI
from llama_index import VectorStoreIndex, ServiceContext, ChatPromptTemplate
from llama_index.core.base_retriever import BaseRetriever
from llama_index.llms import ChatMessage, MessageRole
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import ResponseMode
from llama_index.vector_stores import ChromaVectorStore

from chatbot.config import Config
from db import DB
from execution_context import ExecutionContext
from null_retriever import NullRetriever


class Prompts(ABC):
    """
    Prompts are loaded from the configuration file and provide template text
    to guide the LLM. Every prompt includes one system prompt and one user
    prompt.
    """
    def __init__(self, config_path: str):
        config = Config.get(config_path)
        self._system_prompt = '\n'.join(config['system'])
        self._user_prompt = '\n'.join(config['user'])

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def user_prompt(self) -> str:
        return self._user_prompt


class Step(ABC):
    """
    The Step class represents a single step in the conversation workflow.

    Each step performs some operation like querying a model, evaluating
    the patient, or generating a response. Steps are executed sequentially
    to drive the conversation.

    Steps take prompts and execution context as dependencies. The prompts
    provide templates for system messages. The context shares state between
    steps.

    Subclasses implement the specific logic.
    """

    def __init__(self,
                 prompts: Prompts,
                 db: DB,
                 execution_context: ExecutionContext):
        self._prompts = prompts
        self._db = db
        self._execution_context = execution_context
        self._query_engine = None

    def query(self, query: str, **kwargs) -> str:
        if self._query_engine is None:
            self._query_engine = self._get_query_engine()
        try:
            response = self._query_engine.query(query)
            self._execution_context.handle(response)
            return response.response
        except httpx.ConnectError as e:
            # common reason: Ollama is not running
            raise NetworkError()

    def _get_query_engine(self) -> RetrieverQueryEngine:
        vector_retriever_chunk = self._get_retriever(collection=Config.get('collection'),
                                                     db=self._db.get_instance(),
                                                     service_context=self._execution_context.get_service_context())
        text_template = self._get_prompt_template(system_prompt=self._prompts.system_prompt,
                                                  user_prompt=self._prompts.user_prompt)
        return RetrieverQueryEngine.from_args(
            vector_retriever_chunk,
            service_context=self._execution_context.get_service_context(),
            verbose=True,
            response_mode=ResponseMode.COMPACT,
            text_qa_template=text_template)

    def _get_retriever(self, collection: str, db: ClientAPI,
                       service_context: ServiceContext) -> BaseRetriever:
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


class KnowledgeEnrichedStep(Step, ABC):
    """
    A Step subclass that responds to queries by incorporating additional
    context from a knowledge base.

    When processing a query, this step first queries the knowledge base for any
    relevant information related to the query terms. It then combines the
    knowledge base results with the response from querying the main model.

    This allows the step to return responses that are more informed and
    context-aware by leveraging unstructured knowledge beyond what is
    contained in the main model alone.
    """

    def __init__(self,
                 prompts: Prompts,
                 db: DB,
                 execution_context: ExecutionContext):
        super().__init__(prompts=prompts, db=db, execution_context=execution_context)

    def _get_retriever(self, collection: str, db: ClientAPI,
                       service_context: ServiceContext) -> BaseRetriever:
        chroma_collection = db.get_or_create_collection(collection)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        vector_store_index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
        return vector_store_index.as_retriever(similarity_top_k=self._db.retrieve_n_chunks)


class NetworkError(Exception):

    def __init__(self):
        super().__init__('Network error: did you start Ollama?')