from typing import Optional, Dict, List
from os import listdir
from os.path import join as os_join
from os.path import isfile, join
import chromadb
from chromadb.api.models import Collection
from llama_index import Document
from llama_index.readers.base import BaseReader
from pathlib import Path

from config import Config


class DbLoader:

    def __init__(self):
        self._db_path = Config.get('dbPath')
        self._source_folder = Config.get('dbLoader.sourceFolder')
        self._source_extension = Config.get('dbLoader.sourceExtension')

    def load_db(self):
        db = chromadb.PersistentClient(path=self._db_path)

        files = [f for f in listdir(self._source_folder) if self._should_load(f)]
        for file in files:
            description = file.split('.')[0]
            chroma_collection = db.get_or_create_collection(description)
            print(f'loading {file} into collection={description}')
            self._load_file(filename=file, chroma_collection=chroma_collection)

    def _should_load(self, filename: str) -> bool:
        return isfile(os_join(self._source_folder, filename)) and filename.endswith(self._source_extension)

    def _load_file(self, filename: str, chroma_collection: Collection):
        file = Path(os_join(self._source_folder, filename))
        nodes = SimpleFileReader().load_data(file, {})
        for idx, node in enumerate(nodes):
            chroma_collection.add(
                documents=[node.text],
                ids=[node.node_id],
                metadatas=[node.metadata]
            )


class SimpleFileReader(BaseReader):

    def __init__(self):
        super().__init__()

    def load_data(
            self, file: Path, metadata: Optional[Dict] = None
    ) -> List[Document]:
        with open(file, 'r') as file:
            content = file.read()
            metadata = metadata if metadata is not None else {}
            metadata['file_name'] = file.name
            return [Document(text=content, metadata=metadata)]


DbLoader().load_db()
