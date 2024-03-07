from llama_index.llms import Ollama

from chatbot.step_chat_generation import ChatGenerationStep
from chatbot.step_diagnosis import DiagnosisGenerationStep
from chatbot.step_final_diagnosis import FinalDiagnosisGenerationStep
from chatbot.step_summary import SummaryGenerationStep
from config import Config
from page import Page

from db import DB
from execution_context import ExecutionContext

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
    step_final_diagnosis = FinalDiagnosisGenerationStep(db=db, execution_context=execution_context)

    Page(step_chat=step_chat,
         step_summary=step_summary,
         step_diagnosis=step_diagnosis,
         step_final_diagnosis=step_final_diagnosis)
