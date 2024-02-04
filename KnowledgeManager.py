from typing import Union, List

from langchain_core.vectorstores import VectorStore
from llama_index import VectorStoreIndex, ServiceContext, BaseCallbackHandler
from llama_index.callbacks import CallbackManager
from llama_index.llms.utils import LLMType
from llama_index.vector_stores.types import BasePydanticVectorStore
from chromadb.api.models.Collection import Collection


class KnowledgeManager:

    def __init__(self,
                 llm: LLMType,
                 vector_store: Union[VectorStore, BasePydanticVectorStore],
                 callbacks: List[BaseCallbackHandler]):
        callback_manager = CallbackManager(callbacks)
        service_context = ServiceContext.from_defaults(llm=llm,
                                                       embed_model="local",
                                                       callback_manager=callback_manager)
        vector_store_index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

        self.query_engine = vector_store_index.as_query_engine()

    def query(self, query: str) -> str:
        response = self.query_engine.query(query)
        return response.response
