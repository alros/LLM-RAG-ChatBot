"""
LLM RAG Chatbot
"""
from llama_index.llms import Ollama
from execution_context import ExecutionContext
from step_chat_generation import ChatGenerationStep
from step_diagnosis import DiagnosisGenerationStep
from step_final_diagnosis import FinalDiagnosisGenerationStep
from step_summary import SummaryGenerationStep
from db import DB
from page import Page
from config import Config

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
