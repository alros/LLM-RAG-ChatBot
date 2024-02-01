from llama_index.callbacks import CBEventType, LlamaDebugHandler

import KnowledgeManager


class ChatAssistant:

    def __init__(self, knowledge_manager: KnowledgeManager, llama_debug: LlamaDebugHandler):
        self.knowledge_manager = knowledge_manager
        self.llama_debug = llama_debug

    def ask(self, query: str):
        response = self.knowledge_manager.query(query)
        self._print_debug()
        return response

    def _print_debug(self):
        event_pairs = self.llama_debug.get_event_pairs(CBEventType.LLM)
        print("\n" + ("=" * 20) + "\n")
        for event_pair in event_pairs:
            print(event_pair[0])
            print(event_pair[1].payload.keys())
            print(event_pair[1].payload["response"])

    def clear(self):
        pass
