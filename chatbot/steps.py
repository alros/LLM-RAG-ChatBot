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
from llama_index.response_synthesizers import ResponseMode, get_response_synthesizer
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
        """
        Initialises the instance

        :param config_path: root of the config containing the `system and
               `user` keys. The keys can be expressed as arrays to simplify
               readability in JSON.
        """
        config = Config.get(config_path)
        self._system_prompt = '\n'.join(config['system'])
        self._user_prompt = '\n'.join(config['user'])

    @property
    def system_prompt(self) -> str:
        """
        Returns the system prompt.
        :return: the prompt.
        """
        return self._system_prompt

    @property
    def user_prompt(self) -> str:
        """
        Returns the user prompt.
        :return: the prompt.
        """
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
        """
        Initialises the instance

        :param prompts: instance of Prompts to be used in the step.
        :param db: instance of the database.
        :param execution_context: instance of the execution context.
        """
        self._prompts = prompts
        self._db = db
        self._execution_context = execution_context
        self._query_engine = None

    def query(self, query: str, **kwargs) -> str | None:
        """
        Processes the query and returns an answer or None

        :param query: the query for the step.
        :param kwargs: step-specific parameters.
        :return: the response or None.
        """
        try:
            response = self._get_query_engine(**kwargs).query(query)
            self._execution_context.handle(response)
            return response.response
        except (httpx.ConnectError, httpx.HTTPStatusError, httpx.ReadTimeout) as e:
            # common reason: Ollama is not running
            raise NetworkError()

    def _get_query_engine(self, **kwargs) -> RetrieverQueryEngine:
        """
        Creates the instance of the query engine with lazy loading.
        :return: a singleton.
        """
        if self._query_engine is None:
            service_context = self._execution_context.get_service_context()
            # this creates the vector retriever (it can be a NullRetriever)
            vector_retriever_chunk = self._get_retriever(collection=Config.get('collection'),
                                                         db=self._db.get_instance(),
                                                         service_context=service_context)
            # this creates the prompt templates
            text_template = self._get_prompt_template(system_prompt=self._prompts.system_prompt,
                                                      user_prompt=self._prompts.user_prompt,
                                                      **kwargs)
            # query engine using the retriever and the prompts
            self._query_engine = RetrieverQueryEngine.from_args(
                vector_retriever_chunk,
                service_context=self._execution_context.get_service_context(),
                verbose=True,
                response_mode=ResponseMode.COMPACT,
                text_qa_template=text_template)
        return self._query_engine

    def _get_retriever(self, collection: str, db: ClientAPI,
                       service_context: ServiceContext) -> BaseRetriever:
        """
        Default retriever: a NullRetriever returning an empty result.
        Subclasses can override this method to use different retrievers.

        :param collection: database collection.
        :param db: database instance.
        :param service_context: current service context.
        :return: the retriever for the instance
        """
        return NullRetriever()

    def _get_prompt_template(self, system_prompt: str, user_prompt: str, **kwargs) -> ChatPromptTemplate:
        """
        Utility method to assemble the template.
        :param system_prompt: system prompt.
        :param user_prompt: user prompt.
        :return: instance of ChatPromptTemplate.
        """
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
        """
        Creates the instance.

        :param prompts: prompts to be used.
        :param db: database instance.
        :param execution_context: execution context instance.
        """
        super().__init__(prompts=prompts, db=db, execution_context=execution_context)

    def _get_retriever(self, collection: str, db: ClientAPI,
                       service_context: ServiceContext) -> BaseRetriever:
        """
        Returns a retriever that fetches documents from the database.

        :param collection: database collection.
        :param db: database instance.
        :param service_context: the service context.
        :return: a BaseRetriever that returns content from the database.
        """
        chroma_collection = db.get_or_create_collection(collection)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        vector_store_index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
        return vector_store_index.as_retriever(similarity_top_k=self._db.retrieve_n_chunks)


class NetworkError(Exception):
    """
    Exception raised when the network is not available.
    """

    def __init__(self):
        """
        Creates the instance
        """
        super().__init__('Network error: did you start Ollama?')
