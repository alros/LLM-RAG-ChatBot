from llama_index.callbacks import CBEventType, LlamaDebugHandler
from llama_index.response.schema import RESPONSE_TYPE

import KnowledgeManager


class ChatAssistant:

    def __init__(self, knowledge_manager: KnowledgeManager, llama_debug: LlamaDebugHandler):
        self.knowledge_manager = knowledge_manager
        self.llama_debug = llama_debug

    def ask(self, query: str) -> str:
        response = self.knowledge_manager.query(query)
        self._print_debug(response)
        return response.response

    def _print_debug(self, response: RESPONSE_TYPE):
        event_pairs = self.llama_debug.get_event_pairs(CBEventType.LLM)
        print("\n" + ("=" * 20) + " RESPONSE " + ("=" * 20) + "\n")
        for node in response.source_nodes:
            print(f'{node.node_id}: score {node.score} - {node.node.metadata["file_name"]}\n\n')
        print("\n" + ("=" * 20) + " /RESPONSE " + ("=" * 20) + "\n")
        print("\n" + ("=" * 20) + " DEBUG " + ("=" * 20) + "\n")
        for event_pair in event_pairs:
            print(event_pair[0])
            print(event_pair[1].payload.keys())
            print(event_pair[1].payload["response"])
        print("\n" + ("=" * 20) + " /DEBUG " + ("=" * 20) + "\n")

    def clear(self):
        pass
