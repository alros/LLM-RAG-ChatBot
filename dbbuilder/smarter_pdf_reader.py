from pathlib import Path
from typing import Optional, List, Dict, Tuple

from llama_index import SimpleDirectoryReader, Document
from llama_index.readers.base import BaseReader
from llmsherpa.readers import LayoutPDFReader


class SmarterPDFReader(BaseReader):

    def __init__(self,
                 llm_sherpa_url: str,
                 key_phrase_ngram_range: Optional[Tuple] = (1, 1),
                 key_phrase_diversity: Optional[float] = 0.9):
        super().__init__()
        self.layout_pdf_reader = LayoutPDFReader(llm_sherpa_url)
        self._key_phrase_ngram_range = key_phrase_ngram_range
        self._key_phrase_diversity = key_phrase_diversity

    def load_data(
            self, file_path: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        doc = self.layout_pdf_reader.read_pdf(file_path.absolute().as_posix())
        docs = []
        for idx, chunk in enumerate(doc.chunks()):
            text = chunk.to_context_text(include_section_info=False)
            metadata = {
                'chunk_index': idx,
                'file_name': file_path.name,
            }
            print(f'adding {file_path.name} chunk {idx} to index')
            docs.append(PDFDocument(text=text, metadata=metadata))
        return docs


class PDFDocument(Document):

    def __init__(self,
                 text: str,
                 metadata: Dict):
        super().__init__(text=text, metadata=metadata)

    def __str__(self) -> str:
        index = self.chunk_index()
        header = self.metadata['header']
        keywords = self._keywords_as_str()
        sep = f'{"-" * 32}'
        large_sep = f'{"=" * 32}'
        return f'{large_sep}\nchunk {index}\n{header}\n{keywords}\n{sep}\n{self.text}\n'

    def chunk_index(self) -> int:
        return self.metadata['chunk_index']

    def _keywords_as_str(self, separator: Optional[str] = ' / '):
        return separator.join([f'({k[0]}, {k[1]})' for k in self.metadata['keywords']])
