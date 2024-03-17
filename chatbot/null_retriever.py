"""
LLM RAG Chatbot
"""
from typing import List
from llama_index import QueryBundle
from llama_index.core.base_retriever import BaseRetriever
from llama_index.schema import NodeWithScore, Document


class NullRetriever(BaseRetriever):
    """
     NullRetriever is a simple retriever that returns an empty result.

     It implements the BaseRetriever interface but always returns a single
     NodeWithScore containing an empty Document. This acts as a placeholder
     retriever that can be used during development or when no primary
     retriever is available/functional. By consistently returning an empty
     result, it prevents errors from occurring downstream.
     """

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        It returns an empty list of nodes with scores.
        :param query_bundle: query.
        :return: an empty list of NodeWithScore.
        """
        node = Document(text='', node_id='', metadata={'file_name': ''})
        return [NodeWithScore(node=node, score=0)]
