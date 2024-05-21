"""
LLM RAG Chatbot
"""
from chromadb import ClientAPI
from llama_index import ServiceContext, VectorStoreIndex, ChatPromptTemplate
from llama_index.core.base_retriever import BaseRetriever
from llama_index.core.llms.types import ChatMessage, MessageRole
from llama_index.vector_stores import ChromaVectorStore

from chatbot.config import Config
from chatbot.db import DB
from chatbot.execution_context import ExecutionContext
from chatbot.steps import Prompts, KnowledgeEnrichedStep


class DiscussionPrompt(Prompts):

    def __init__(self):
        super().__init__('prompts.discussion')


class DiscussionStep(KnowledgeEnrichedStep):

    def __init__(self, db: DB, execution_context: ExecutionContext):
        super().__init__(prompts=DiscussionPrompt(), db=db, execution_context=execution_context)
        self._retrieve_N_chunks = 5

    def query(self, query: str, **kwargs) -> str:
        summary = kwargs['summary']
        next_answer = super().query(query, summary=summary)
        return next_answer

    def _get_retriever(self, collection: str, db: ClientAPI,
                       service_context: ServiceContext) -> BaseRetriever:
        db_collection = Config.get('collection') + Config.get('dbLoader.kbCollectionSuffix')
        chroma_collection = db.get_or_create_collection(db_collection)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        vector_store_index = VectorStoreIndex.from_vector_store(vector_store,
                                                                service_context=service_context)
        return vector_store_index.as_retriever(similarity_top_k=self._retrieve_N_chunks)  # best N chunks

    def _get_prompt_template(self, system_prompt: str, user_prompt: str, **kwargs) -> ChatPromptTemplate:
        summary = kwargs['summary']
        chat_text_qa_msgs = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt.format(summary)
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=user_prompt,
            ),
        ]
        return ChatPromptTemplate(chat_text_qa_msgs)
