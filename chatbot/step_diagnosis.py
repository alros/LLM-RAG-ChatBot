from chatbot.db import DB
from chatbot.execution_context import ExecutionContext
from chatbot.steps import Step, Prompts, KnowledgeEnrichedStep


class DiagnosisPrompt(Prompts):
    def __init__(self):
        super().__init__('prompts.diagnosis')


class DiagnosisGenerationStep(KnowledgeEnrichedStep):
    def __init__(self, db: DB, execution_context: ExecutionContext):
        super().__init__(prompts=DiagnosisPrompt(), db=db, execution_context=execution_context)
