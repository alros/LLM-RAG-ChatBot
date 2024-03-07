from typing import Tuple

import streamlit as st
from streamlit_chat import message

from steps import Step


class Page:
    SESSION_MESSAGES = 'messages'
    USER_INPUT = 'user_input'
    SPINNER_THINKING = 'thinking_spinner'
    FINAL_DIAGNOSIS = 'final_diagnosis'

    def __init__(self,
                 step_chat: Step,
                 step_summary: Step,
                 step_diagnosis: Step,
                 step_final_diagnosis: Step):
        self._step_chat = step_chat
        self._step_summary = step_summary
        self._step_diagnosis = step_diagnosis
        self._step_final_diagnosis = step_final_diagnosis

        st.set_page_config(page_title='Medi-chat')
        if Page.SESSION_MESSAGES not in st.session_state:
            st.session_state[Page.SESSION_MESSAGES] = []

        st.header('Chatbot')
        st.subheader('Assistant')

        if Page.FINAL_DIAGNOSIS in st.session_state and st.session_state[Page.FINAL_DIAGNOSIS] is not None:
            self._close_session(st.session_state[Page.FINAL_DIAGNOSIS])

        else:
            self._continue_the_chat()

    def _continue_the_chat(self):
        next_question = self._get_next_question()
        st.session_state[Page.SESSION_MESSAGES].append({
            'q': next_question
        })

        self._update_the_page()

        st.text_input('Your reply', key=Page.USER_INPUT, on_change=self._process_input)

    def _close_session(self, final_diagnosis: str):
        st.session_state[Page.SESSION_MESSAGES].append({
            'q': final_diagnosis
        })
        self._update_the_page()

    def _update_the_page(self):
        for i, msg in enumerate(st.session_state[Page.SESSION_MESSAGES]):
            message(msg['q'], is_user=False, key=str(i) + 'q')
            if 'a' in msg:
                message(msg['a'], is_user=True, key=str(i) + 'a')

        st.session_state[Page.SPINNER_THINKING] = st.empty()

    def _get_next_question(self):
        _, query = self._get_chat()
        return self._step_chat.query(query)

    def _get_chat(self) -> Tuple[int, str]:
        messages = st.session_state[Page.SESSION_MESSAGES]
        chat = ''
        for qa in messages:
            chat = chat + '\n' if chat else ''
            chat = f'{chat}You: "{qa["q"]}"'
            chat = f'{chat}\nPatient: "{qa["a"]}"' if 'a' in qa else chat
        return len(messages), chat

    # def _get_summary(self):
    #     _, query = self._get_chat()
    #     return self._step_summary.query(query)

    def _process_input(self):
        if st.session_state[Page.USER_INPUT] and len(st.session_state[Page.USER_INPUT].strip()) > 0:
            self._process_user_text(st.session_state[Page.USER_INPUT].strip())

    def _process_user_text(self, user_text: str):
        st.session_state[Page.SESSION_MESSAGES][len(st.session_state[Page.SESSION_MESSAGES]) - 1]['a'] = user_text
        with st.session_state[Page.SPINNER_THINKING], st.spinner('Thinking'):
            number_of_questions, chat = self._get_chat()
            summary = self._step_summary.query(chat)
            diagnostic = self._step_diagnosis.query(summary)
            final_diagnosis = self._step_final_diagnosis.query(diagnostic, number_of_questions=number_of_questions)
            st.session_state[Page.FINAL_DIAGNOSIS] = final_diagnosis
            st.session_state[Page.USER_INPUT] = ''
