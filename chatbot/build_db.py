from typing import List

import chromadb
from llama_index import SimpleDirectoryReader
from llama_index.callbacks import LlamaDebugHandler
from llama_index.extractors import TitleExtractor
from llama_index.ingestion import IngestionPipeline
from llama_index.llms import Ollama
from llama_index.node_parser import SentenceSplitter
from llama_index.schema import MetadataMode, TextNode
from llama_index.vector_stores import ChromaVectorStore

from chatbot.config import Config

db = chromadb.PersistentClient(path=Config.get('dbPath'))
chroma_collection = db.get_or_create_collection(Config.get('collection'))
store = ChromaVectorStore(chroma_collection=chroma_collection)

llm = Ollama(model=Config.get('model'))

llama_debug = LlamaDebugHandler(print_trace_on_end=True)

transformers = [
    SentenceSplitter(chunk_size=1024, chunk_overlap=20),
    TitleExtractor(
        llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8
    )
]
pipeline = IngestionPipeline(transformations=transformers)
reader = SimpleDirectoryReader(Config.get('kb'), recursive=True)

for docs in reader.iter_data():
    base_nodes: List[TextNode] = pipeline.run(documents=docs)

    for idx, node in enumerate(base_nodes):
        print(f'adding {node.metadata["file_path"]} / {idx} / {node.node_id}')
        chroma_collection.add(
            documents=[node.text],
            ids=[node.node_id],
            metadatas=[node.metadata]
        )
