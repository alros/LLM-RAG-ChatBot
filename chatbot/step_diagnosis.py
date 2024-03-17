"""
LLM RAG Chatbot
"""
from chatbot.db import DB
from chatbot.execution_context import ExecutionContext
from chatbot.steps import Prompts, KnowledgeEnrichedStep


class DiagnosisPrompt(Prompts):
    """
    The DiagnosisPrompt class represents a set of prompt templates
    for generating the initial evaluation of the patient.
    """

    def __init__(self):
        """
        Creates the instance using the prompts.diagnosis configuration.
        """
        super().__init__('prompts.diagnosis')


class DiagnosisGenerationStep(KnowledgeEnrichedStep):
    """
    The DiagnosisGenerationStep class represents the step to generate
    the initial evaluation of the patient. The output in JSON is then
    evaluated to define the final diagnosis in another step.
    """

    def __init__(self, db: DB, execution_context: ExecutionContext):
        """
        Creates the instance.
        :param db: database instance.
        :param execution_context: execution context instance.
        """
        super().__init__(prompts=DiagnosisPrompt(), db=db, execution_context=execution_context)
