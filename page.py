import streamlit as st
from streamlit_chat import message

from steps import Step


class Page:
    SESSION_MESSAGES = 'messages'
    USER_INPUT = 'user_input'
    SPINNER_THINKING = 'thinking_spinner'

    def __init__(self,
                 step_chat: Step,
                 step_summary: Step,
                 step_diagnosis: Step):
        self._step_chat = step_chat
        self._step_summary = step_summary
        self._step_diagnosis = step_diagnosis

        st.set_page_config(page_title='Medi-chat')
        if Page.SESSION_MESSAGES not in st.session_state:
            st.session_state[Page.SESSION_MESSAGES] = []

        st.header('Chatbot')
        st.subheader('Assistant')

        next_question = self._get_next_question()
        st.session_state[Page.SESSION_MESSAGES].append({
            'q': next_question
        })

        for i, msg in enumerate(st.session_state[Page.SESSION_MESSAGES]):
            message(msg['q'], is_user=False, key=str(i) + 'q')
            if 'a' in msg:
                message(msg['a'], is_user=True, key=str(i) + 'a')

        st.session_state[Page.SPINNER_THINKING] = st.empty()

        st.text_input('Your reply', key=Page.USER_INPUT, on_change=self._process_input)

    def _get_next_question(self):
        query = self._get_chat()
        return self._step_chat.query(query)

    def _get_chat(self):
        messages = st.session_state[Page.SESSION_MESSAGES]
        chat = ''
        for qa in messages:
            chat = chat + '\n' if chat else ''
            chat = f'{chat}You: "{qa["q"]}"'
            chat = f'{chat}\nPatient: "{qa["a"]}"' if 'a' in qa else chat
        return chat

    def _get_summary(self):
        query = self._get_chat()
        return self._step_summary.query(query)

    def _process_input(self):
        if st.session_state[Page.USER_INPUT] and len(st.session_state[Page.USER_INPUT].strip()) > 0:
            self._process_user_text(st.session_state[Page.USER_INPUT].strip())

    def _process_user_text(self, user_text: str):
        st.session_state[Page.SESSION_MESSAGES][len(st.session_state[Page.SESSION_MESSAGES]) - 1]['a'] = user_text
        with st.session_state[Page.SPINNER_THINKING], st.spinner('Thinking'):
            chat = self._get_chat()
            summary = self._step_summary.query(chat)
            diagnostic = self._step_diagnosis.query(summary)
            print(f'diagnostic: ' + diagnostic)
            st.session_state[Page.USER_INPUT] = ''
