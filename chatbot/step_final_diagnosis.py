"""
LLM RAG Chatbot
"""
import json

from chatbot.config import Config
from chatbot.db import DB
from chatbot.execution_context import ExecutionContext
from chatbot.steps import Step, Prompts, KnowledgeEnrichedStep


class FinalDiagnosisPrompts(Prompts):
    """
    The FinalDiagnosisPrompts class represents a set of prompt templates
    for generating the final diagnosis of a patient.
    """
    def __init__(self):
        """
        Creates the instance using the prompts.final_diagnosis configuration.
        """
        super().__init__('prompts.final_diagnosis')


class FinalDiagnosisGenerationStep(KnowledgeEnrichedStep):
    """
    The FinalDiagnosisGenerationStep class represents the step to generate
    the final diagnosis of a patient.
    """

    def __init__(self, db: DB, execution_context: ExecutionContext):
        super().__init__(prompts=FinalDiagnosisPrompts(), db=db, execution_context=execution_context)

    def query(self, query: str, **kwargs) -> str:
        final_diagnosis = self._get_diagnosis(query, kwargs['number_of_questions'])
        return super().query(final_diagnosis) if final_diagnosis else None

    def _get_diagnosis(self, response: str, number_of_questions: int = -1) -> str | None:
        if number_of_questions < Config.get('diagnosis.minimum_number_of_questions'):
            return None
        document = json.loads(response)
        number_of_symptoms = document['Number']
        severity = document['Severity']
        confidence = document['Confidence']

        if number_of_questions > Config.get('diagnosis.maximum_number_of_questions'):
            return Config.get('diagnosis.negative_diagnosis') + document['Explanation']

        if confidence < Config.get('diagnosis.minimum_confidence'):
            return None

        if number_of_symptoms < Config.get('diagnosis.minimum_number_of_symptoms'):
            return None

        score = severity * number_of_symptoms * confidence

        if score < Config.get('diagnosis.minimum_score'):
            return None

        return document['Explanation']
