"""
LLM RAG Chatbot
"""
import json

from chatbot.db import DB
from chatbot.execution_context import ExecutionContext
from chatbot.steps import Step, Prompts, KnowledgeEnrichedStep


class ChatPrompts(Prompts):
    def __init__(self):
        super().__init__('prompts.chat')


class ChatGenerationStep(KnowledgeEnrichedStep):
    def __init__(self, db: DB, execution_context: ExecutionContext):
        super().__init__(prompts=ChatPrompts(), db=db, execution_context=execution_context)

    def query(self, query: str, **kwargs):
        if not query:
            return 'How old are you?'
        next_question = super().query(query)
        return json.loads(next_question)['Question']
