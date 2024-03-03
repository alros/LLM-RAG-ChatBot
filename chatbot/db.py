import chromadb
from chromadb import ClientAPI


class DB:
    def __init__(self, db_path: str, collection: str, retrieve_n_chunks: int = 1):
        self._db_path = db_path
        self._collection = collection
        self._retrieve_n_chunks = retrieve_n_chunks
        self._db_instance = None

    def get_instance(self) -> ClientAPI:
        if self._db_instance is None:
            self._db_instance = chromadb.PersistentClient(path=self._db_path)
        return self._db_instance

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def retrieve_n_chunks(self) -> int:
        return self._retrieve_n_chunks
