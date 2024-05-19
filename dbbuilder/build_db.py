"""
LLM RAG Chatbot
"""
from os import listdir
from os.path import isfile
from os.path import join as os_join
from pathlib import Path

import chromadb
from chromadb.api.models import Collection
from urllib3.exceptions import MaxRetryError

from chatbot.config import Config
from dbbuilder.simple_file_reader import SimpleFileReader
from dbbuilder.smarter_pdf_reader import SmarterPDFReader


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

        The filename without extension determines the name of the target database
        collection: 1 file 1 collection

        :return: None.
        """

        # creates an instance of Chroma
        db = chromadb.PersistentClient(path=self._db_path)

        # iterates over the files in the folder
        # only files with the right extension are loaded
        files = [f for f in listdir(self._source_folder) if self._should_load(f)]
        for file in files:
            # The filename determines the collection name
            extension = file.split('.')[1]

            base_chroma_collection = Config.get('collection')
            if extension == 'pdf':
                collection_name = base_chroma_collection + Config.get('dbLoader.kbCollectionSuffix')
                chroma_collection = db.get_or_create_collection(collection_name)
                print(f'loading {file} into collection={collection_name}')
                try:
                    self._load_file_kb(filename=file, chroma_collection=chroma_collection)
                except MaxRetryError as e:
                    print('Connection failed, please verify that NLM-Ingestor runs on the expected port.')
                    print('Execution aborted.')
                    break
            else:
                collection_name = base_chroma_collection
                chroma_collection = db.get_or_create_collection(collection_name)
                print(f'loading {file} into collection={collection_name}')
                self._load_file(filename=file, chroma_collection=chroma_collection)

    def _should_load(self, filename: str) -> bool:
        """
        Check if the file should be loaded into the db.

        :param filename: filename in the folder
        :return: True if filename is a file and it has the correct extension.
        """
        return (isfile(os_join(self._source_folder, filename))
                and (filename.endswith(self._source_extension)
                     or filename.endswith('pdf')))

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

    def _load_file_kb(self, filename: str, chroma_collection: Collection) -> None:
        """
        Load the file into the db.
        :param filename: filename to load.
        :param chroma_collection: chroma db collection reference.
        :return: None
        """

        file = Path(os_join(self._source_folder, filename))
        reader = SmarterPDFReader(Config.get('dbLoader.llmSherpaUrl'))
        chunks = reader.load_data(file_path=file)
        for chunk in chunks:
            print(f'\n\n{chunk}\n\n')
            chroma_collection.add(
                documents=[chunk.text],
                ids=[chunk.node_id],
                metadatas=[chunk.metadata]
            )


# This adds the document to the db. It creates the db if needed
DbLoader().load_db()
