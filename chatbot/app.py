from llama_index.legacy.llms import Ollama

from config import Config
from page import Page

from db import DB
from execution_context import ExecutionContext
from steps import ChatGenerationStep, SummaryGenerationStep, DiagnosisGenerationStep

if __name__ == "__main__":
    collection = Config.get('collection')
    db_path = Config.get('dbPath')
    model = Config.get('model')

    db = DB(db_path=db_path,
            collection=collection)
    llm = Ollama(model=model)

    execution_context = ExecutionContext(llm=llm)
    step_chat = ChatGenerationStep(db=db, execution_context=execution_context)
    step_summary = SummaryGenerationStep(db=db, execution_context=execution_context)
    step_diagnosis = DiagnosisGenerationStep(db=db, execution_context=execution_context)

    Page(step_chat=step_chat, step_summary=step_summary, step_diagnosis=step_diagnosis)
