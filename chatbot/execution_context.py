from llama_index import ServiceContext
from llama_index.callbacks import LlamaDebugHandler, CallbackManager, CBEventType
from llama_index.llms import CustomLLM
from llama_index.response.schema import RESPONSE_TYPE


class ExecutionContext:

    def __init__(self, llm: CustomLLM):
        self._llm = llm
        self._service_context = None
        self._llama_debug = None

    def get_service_context(self):
        if self._service_context is None:
            self._llama_debug = LlamaDebugHandler(print_trace_on_end=True)
            callback_manager = CallbackManager([self._llama_debug])
            self._service_context = ServiceContext.from_defaults(llm=self._llm,
                                                                 embed_model="local",
                                                                 callback_manager=callback_manager)
        return self._service_context

    def handle(self, response: RESPONSE_TYPE):
        self._print_debug(response=response, llama_debug=self._llama_debug)

    def _print_debug(self, llama_debug: LlamaDebugHandler, response):
        event_pairs = llama_debug.get_event_pairs(CBEventType.LLM)
        print('\n==================== RESPONSE ====================\n')
        print('\n  ------------------ source nodes ----------------')
        for node in response.source_nodes:
            print(f'  {node.node_id}: score {node.score} - {node.node.metadata["file_name"]}\n\n')
        print('  ----------------- /source nodes ----------------\n')
        print('\n  ------------------ events pairs ----------------\n')
        for event_pair in event_pairs:
            print(f'  {event_pair[0]}')
            print(f'  {event_pair[1].payload.keys()}')
            print(f'  {event_pair[1].payload["response"]}')
        print('  ------------------ events pairs ----------------\n')
        print('\n=================== /RESPONSE ====================\n')
