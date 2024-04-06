"""
LLM RAG Chatbot
"""
from chatbot.config import Config
from chatbot.step_discuss import DiscussionStep
from execution_context import ExecutionContext
from step_chat_generation import ChatGenerationStep
from step_diagnosis import DiagnosisGenerationStep
from step_final_diagnosis import FinalDiagnosisGenerationStep
from step_summary import SummaryGenerationStep
from db import DB
from page import Page

#
# This is the main script to run the LLM RAG Chatbot.
#

if __name__ == "__main__":
    # assemble the dependencies
    db = DB(Config.get('collection'))
    db_kb = DB(f'{Config.get("collection")}{Config.get("dbLoader.kbCollectionSuffix")}')
    execution_context = ExecutionContext()
    step_chat = ChatGenerationStep(db=db, execution_context=execution_context)
    step_summary = SummaryGenerationStep(db=db, execution_context=execution_context)
    step_diagnosis = DiagnosisGenerationStep(db=db, execution_context=execution_context)
    step_final_diagnosis = FinalDiagnosisGenerationStep(db=db, execution_context=execution_context)
    step_discussion = DiscussionStep(db=db_kb, execution_context=execution_context)

    # wire the dependencies into the streamlit Page
    Page(step_chat=step_chat,
         step_summary=step_summary,
         step_diagnosis=step_diagnosis,
         step_final_diagnosis=step_final_diagnosis,
         step_discussion=step_discussion)
