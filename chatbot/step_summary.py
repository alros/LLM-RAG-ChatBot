"""
LLM RAG Chatbot
"""
from chatbot.db import DB
from chatbot.execution_context import ExecutionContext
from chatbot.steps import Step, Prompts


class SummaryPrompts(Prompts):
    """
    The SummaryPrompts class represents a set of prompt templates for
    generating the summaries of the dialogues.
    """
    def __init__(self):
        super().__init__('prompts.summary')


class SummaryGenerationStep(Step):
    """
    The SummaryGenerationStep class represents the step to generate
    the summaries of the dialogues.
    """
    def __init__(self, db: DB, execution_context: ExecutionContext):
        super().__init__(prompts=SummaryPrompts(), db=db, execution_context=execution_context)
