from pathlib import Path
from typing import Optional, List, Dict
from llama_index import Document
from llama_index.readers.base import BaseReader


class SimpleFileReader(BaseReader):
    """
    Utility class to load the data from disk and return it as a Document
    """

    def __init__(self):
        """
        Builds the instance.
        """
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
