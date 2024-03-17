"""
LLM RAG Chatbot
"""
import json
from chatbot.config import Config
from chatbot.db import DB
from chatbot.execution_context import ExecutionContext
from chatbot.steps import Prompts, KnowledgeEnrichedStep


class ChatPrompts(Prompts):
    """
    The ChatPrompts class represents a set of prompt templates
    for generating next question.
    """

    def __init__(self):
        """
        Creates the instance using the prompts.chat configuration.
        """
        super().__init__('prompts.chat')


class ChatGenerationStep(KnowledgeEnrichedStep):
    """
    The ChatGenerationStep class represents the step to generate
    the questions to the patient.
    """

    def __init__(self, db: DB, execution_context: ExecutionContext):
        """
        Creates the instance
        :param db: reference to the database.
        :param execution_context: reference to the Execution Context.
        """
        super().__init__(prompts=ChatPrompts(), db=db, execution_context=execution_context)

    def query(self, query: str, **kwargs) -> str:
        """
        Queries the LLM to generate the next question using RAG.

        :param query: the current dialogue.
        :param kwargs: not used.
        :return: next question for the patient.
        """
        if not query:
            return Config.get('chat.defaultQuestion')
        next_question = super().query(query)
        return json.loads(next_question)['Question']
