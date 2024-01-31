import chromadb
from llama_index.callbacks import LlamaDebugHandler
from llama_index.llms import Ollama
from llama_index.vector_stores import ChromaVectorStore

from ChatAssistant import ChatAssistant
from KnowledgeManager import KnowledgeManager
from Page import Page


def new_knowledge_manager():
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("bio")
    store = ChromaVectorStore(chroma_collection=chroma_collection)

    mistral = Ollama(model="mistral")

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)

    return KnowledgeManager(kb_path="kb",
                            vector_store=store,
                            llm=mistral,
                            callbacks=[llama_debug])


if __name__ == "__main__":
    km = new_knowledge_manager()
    ca = ChatAssistant(knowledge_manager=km)
    Page(chatAssistant=ca)
