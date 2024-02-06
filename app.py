import chromadb
from llama_index.callbacks import LlamaDebugHandler
from llama_index.llms import Ollama
from llama_index.vector_stores import ChromaVectorStore

from ChatAssistant import ChatAssistant
from KnowledgeManager import KnowledgeManager
from Page import Page

from constants import *


def new_knowledge_manager():
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(db_collection)
    store = ChromaVectorStore(chroma_collection=chroma_collection)

    mistral = Ollama(model=model)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)

    return (KnowledgeManager(vector_store=store,
                             llm=mistral,
                             callbacks=[llama_debug]),
            llama_debug)


if __name__ == "__main__":
    km, llama_debug = new_knowledge_manager()
    ca = ChatAssistant(knowledge_manager=km, llama_debug=llama_debug)
    Page(chatAssistant=ca)
