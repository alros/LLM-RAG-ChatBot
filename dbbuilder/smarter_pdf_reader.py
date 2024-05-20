import re
import statistics
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
        self._min_block_len = 1000
        self._min_block_len_for_class = 100

    def load_data(self, file_path: Path) -> List[Document]:
        """
        It ignores tables, and single lines. It stops once it reaches the References block

        :param file_path: file to parse
        :return: list of chunks
        """
        doc = self.layout_pdf_reader.read_pdf(file_path.absolute().as_posix())
        docs = []
        all_chunks = doc.chunks()
        base_block_class = self._find_base_block_class(all_chunks)
        for idx, chunk in enumerate(all_chunks):
            header = chunk.to_context_text(include_section_info=True).split('\n')[0]
            if chunk.tag == 'table':
                continue
            if header.startswith("References") or header.find('> References') >= 0:
                break
            # if not base_block_class:
            #     base_block_class = chunk.block_json['block_class']
            if chunk.block_json['block_class'] != base_block_class:
                continue

            text = chunk.to_context_text(include_section_info=True)
            metadata = {
                'parent_text': chunk.parent_text(),
                'chunk_index': idx,
                'file_name': file_path.name
            }
            docs.append(PDFDocument(text=text, metadata=metadata))
        docs = self._compact(docs)
        docs = self._fix_newlines_and_citations(docs)
        return docs

    def _find_base_block_class(self, chunks: List[Document]) -> str:
        """
        finds the class of the longest block, presumably a text block and not a title or a citation
        :param chunks:
        :return:
        """
        max_len = -1
        base_block_class = None
        for chunk in chunks:
            text = chunk.to_context_text(include_section_info=False)
            mean_str_len = statistics.mean([len(line) for line in text.split('\n')])
            if len(text) > max_len and mean_str_len > self._min_block_len_for_class and chunk.tag != 'table':
                base_block_class = chunk.block_json['block_class']
                max_len = len(text)
        return base_block_class

    def _compact(self, docs: List[Document]) -> List[Document]:
        """
        chain paragraphs with the same header

        :param docs: list of docs
        :return: compacted the list of docs
        """

        def init_current_doc(doc: Document, idx: int) -> (Document, int, List[int], str):
            original_chunks = doc.metadata['chunk_index']
            doc.metadata['original_chunks'] = str(original_chunks)
            doc.metadata['chunk_index'] = idx
            return doc, idx, [original_chunks], doc.text.split('\n')[0]

        final = []
        if len(docs) == 0:
            return final
        current, idx, original_chunks, to_remove = init_current_doc(docs[0], docs[0].metadata['chunk_index'])
        for i in range(1, len(docs)):
            if ((current.metadata['parent_text'] == docs[i].metadata['parent_text'] and
                 current.metadata['file_name'] == docs[i].metadata['file_name']) or
                    len(current.text) < self._min_block_len):
                original_chunks.append(docs[i].metadata['chunk_index'])
                metadata = {
                    'parent_text': current.metadata['parent_text'],
                    'chunk_index': idx,
                    'file_name': current.metadata['file_name'],
                    'original_chunks': ','.join(map(str, original_chunks))
                }
                text_to_append = docs[i].text[len(to_remove) + 1:]
                current = PDFDocument(text='\n'.join([current.text, text_to_append]), metadata=metadata)
            else:
                final.append(current)
                current, idx, original_chunks, to_remove = init_current_doc(docs[i], idx + 1)
        final.append(current)
        return final

    def _fix_newlines_and_citations(self, docs: List[Document]):
        """
        removes breaking - inside a word caused by new lines in the original PDF
        :param docs: list to filter
        :return: the same list with a cleaned text
        """
        for _, doc in enumerate(docs):
            doc.text = re.sub(r"(\w)- (\w)", r"\1\2", doc.text)
            doc.text = re.sub(r" (\[[\d, ]+])|(\([\d, ]+\)) ", r" ", doc.text)
        return docs


class PDFDocument(Document):
    """
    Represents a PDF Document
    """

    def __init__(self,
                 text: str,
                 metadata: Dict):
        super().__init__(text=text, metadata=metadata)

    def __str__(self):
        return self.text
