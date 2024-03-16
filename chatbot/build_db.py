"""
LLM RAG Chatbot
"""
from typing import Optional, Dict, List
from os import listdir
from os.path import join as os_join
from os.path import isfile
import chromadb
from chromadb.api.models import Collection
from llama_index import Document
from llama_index.readers.base import BaseReader
from pathlib import Path

from config import Config


class DbLoader:
    """
    The DbLoader class provides functionality for loading data inside a
    chroma db.
    """

    def __init__(self):
        """
        Initialize the DbLoader object.

        The initialisation is done by fetching the configuration from
        the config file.
        """
        self._db_path = Config.get('dbPath')
        self._source_folder = Config.get('dbLoader.sourceFolder')
        self._source_extension = Config.get('dbLoader.sourceExtension')

    def load_db(self) -> None:
        """
        Load the data from the source folder into the db.

        The path of the database is specified in config.json 'dbPath'.
        The source folder is specified in config.json 'dbLoader.sourceFolder'.
        The source file extension is specified in config.json 'dbLoader.sourceExtension'.

        :return: None.
        """
        db = chromadb.PersistentClient(path=self._db_path)

        files = [f for f in listdir(self._source_folder) if self._should_load(f)]
        for file in files:
            description = file.split('.')[0]
            chroma_collection = db.get_or_create_collection(description)
            print(f'loading {file} into collection={description}')
            self._load_file(filename=file, chroma_collection=chroma_collection)

    def _should_load(self, filename: str) -> bool:
        """
        Check if the file should be loaded into the db.

        :param filename: filename in the folder
        :return: True if filename is a file and it has the correct extension.
        """
        return (isfile(os_join(self._source_folder, filename))
                and filename.endswith(self._source_extension))

    def _load_file(self, filename: str, chroma_collection: Collection) -> None:
        """
        Load the file into the db.
        :param filename: filename to load.
        :param chroma_collection: chroma db collection reference.
        :return: None
        """
        file = Path(os_join(self._source_folder, filename))
        nodes = SimpleFileReader().load_data(file, {})
        for _, node in enumerate(nodes):
            chroma_collection.add(
                documents=[node.text],
                ids=[node.node_id],
                metadatas=[node.metadata]
            )


class SimpleFileReader(BaseReader):
    """
    Utility class to load the data from disk and return it as a Document
    """

    def __init__(self):
        super().__init__()

    def load_data(
            self, file: Path, metadata: Optional[Dict] = None
    ) -> List[Document]:
        """
        Reads a file returning a list containing a single Document.

        :param file: Path to the file.
        :param metadata: optional metadata.
        :return: a list of length 1 with a Document containing the
                file.
        """
        with open(file, 'r') as file:
            content = file.read()
            metadata = metadata if metadata is not None else {}
            metadata['file_name'] = file.name
            return [Document(text=content, metadata=metadata)]


DbLoader().load_db()
