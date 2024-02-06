from typing import Union, List

from langchain_core.vectorstores import VectorStore
from llama_index import VectorStoreIndex, ServiceContext, BaseCallbackHandler
from llama_index.callbacks import CallbackManager
from llama_index.llms.utils import LLMType
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.response_synthesizers import ResponseMode
from llama_index.vector_stores.types import BasePydanticVectorStore
from constants import *


class KnowledgeManager:

    def __init__(self,
                 llm: LLMType,
                 vector_store: Union[VectorStore, BasePydanticVectorStore],
                 callbacks: List[BaseCallbackHandler]):
        callback_manager = CallbackManager(callbacks)
        service_context = ServiceContext.from_defaults(llm=llm,
                                                       embed_model="local",
                                                       callback_manager=callback_manager)
        vector_store_index = VectorStoreIndex.from_vector_store(vector_store,
                                                                service_context=service_context)
        vector_retriever_chunk = vector_store_index.as_retriever(similarity_top_k=retrieve_N_chunks)  # best N chunks

        # self.query_engine = vector_store_index.as_query_engine()
        self.query_engine = RetrieverQueryEngine.from_args(
            vector_retriever_chunk,
            service_context=service_context,
            verbose=True,
            response_mode=ResponseMode.COMPACT
        )

    def query(self, query: str) -> RESPONSE_TYPE:
        response = self.query_engine.query(query, where={"file_name": "ParadiseLost.txt"})
        return response
