from chatbot.db import DB
from chatbot.execution_context import ExecutionContext
from chatbot.steps import Step, Prompts


class SummaryPrompts(Prompts):
    def __init__(self):
        super().__init__('prompts.summary')


class SummaryGenerationStep(Step):
    def __init__(self, db: DB, execution_context: ExecutionContext):
        super().__init__(prompts=SummaryPrompts(), db=db, execution_context=execution_context)
