"""
LLM RAG Chatbot
"""
import chromadb
from chromadb import ClientAPI

from config import Config


class DB:
    """
    The DB class provides a wrapper to hold the reference to the embedded chroma
    database.
    """

    def __init__(self, collection_name:str):
        """
        Initialise the instance. The configuration is fetched from `config.yaml`.
        """

        self._db_path = Config.get('dbPath')

        # This is something for future evolutions where a collection can contain
        # multiple documents split in multiple chunks
        self._retrieve_n_chunks = 1

        self._db_instance = None

    def get_instance(self) -> ClientAPI:
        """
        Retrieve the instance of the Chroma database.

        This method uses lazy-loading.
        :return: a singleton instance of the Chroma database.
        """
        if self._db_instance is None:
            self._db_instance = chromadb.PersistentClient(path=self._db_path)
        return self._db_instance

    @property
    def retrieve_n_chunks(self) -> int:
        """
        Retrieve the number of chunks to retrieve.

        :return: the number of chunks to retrieve.
        """
        return self._retrieve_n_chunks
