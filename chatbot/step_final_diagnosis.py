"""
LLM RAG Chatbot
"""
import json
from chatbot.config import Config
from chatbot.db import DB
from chatbot.execution_context import ExecutionContext
from chatbot.steps import Prompts, KnowledgeEnrichedStep


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
        """
        Initialise the instance.

        :param db: the chroma database
        :param execution_context: the execution context.
        """
        super().__init__(prompts=FinalDiagnosisPrompts(), db=db, execution_context=execution_context)

    def query(self, query: str, **kwargs) -> str | None:
        """
        Translates the diagnosis in 3rd person into a final exchange in 2nd person.
        It returns None if this is not possible

        :param query: current diagnosis in JSON.
        :param kwargs: number_of_questions = the number of questions.
        :return: final message for the patient or None.
        """
        final_diagnosis = self._get_diagnosis(query, kwargs['number_of_questions'])
        return super().query(final_diagnosis) if final_diagnosis else None

    def _get_diagnosis(self, response: str, number_of_questions: int = -1) -> str | None:
        """
        Formulate the final diagnosis or return None if it is not possible.

        :param response: text with the LLM's diagnosis.
        :param number_of_questions: how many questions were asked to reach this point.
        :return: the final diagnosis or None.
        """

        if number_of_questions < Config.get('diagnosis.minimum_number_of_questions'):
            # ignore, if there were too few questions.
            return None
        document = json.loads(response)
        number_of_symptoms = document['Number']

        if number_of_questions > Config.get('diagnosis.maximum_number_of_questions'):
            # if there were too many questions, return a negative / inconclusive diagnosis
            return Config.get('diagnosis.negative_diagnosis') + document['Explanation']

        confidence = document['Confidence']
        if confidence < Config.get('diagnosis.minimum_confidence'):
            # if the minimum confidence is not met, return None.
            return None

        if number_of_symptoms < Config.get('diagnosis.minimum_number_of_symptoms'):
            # if the minimum number of symptoms is not met, return None.
            return None

        severity = document['Severity']
        score = severity * number_of_symptoms * confidence
        if score < Config.get('diagnosis.minimum_score'):
            # if the minimum score is not met, return None.
            return None

        # Positive diagnosis, return it as a result.
        # This diagnosis talks about the patient in 3rd person and must be
        # transformed into a dialog
        return document['Explanation']
