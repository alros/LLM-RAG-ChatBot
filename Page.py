import streamlit as st
from streamlit_chat import message

from ChatAssistant import ChatAssistant
from constants import *


class Page:

    def __init__(self, chatAssistant: ChatAssistant):
        st.set_page_config(page_title=TEXT_TITLE)
        if SESSION_MESSAGES not in st.session_state:
            st.session_state['messages'] = []
        if SESSION_ASSISTANT not in st.session_state:
            st.session_state[SESSION_ASSISTANT] = chatAssistant

        st.header(TEXT_HEADER)

        st.subheader(TEXT_SUBHEADER)
        for i, (msg, is_user) in enumerate(st.session_state[SESSION_MESSAGES]):
            message(msg, is_user=is_user, key=str(i))
        st.session_state[SESSION_SPINNER_THINKING] = st.empty()

        st.text_input(INPUT_MESSAGE, key=INPUT_USER, on_change=self._process_input)

    def _process_input(self):
        if st.session_state[INPUT_USER] and len(st.session_state[INPUT_USER].strip()) > 0:
            self._process_user_text(st.session_state[INPUT_USER].strip())

    def _process_user_text(self, user_text: str):
        with st.session_state[SESSION_SPINNER_THINKING], st.spinner(TEXT_THINKING):
            assistant_reply = st.session_state[SESSION_ASSISTANT].ask(user_text)

        st.session_state[SESSION_MESSAGES].append((user_text, True))
        st.session_state[SESSION_MESSAGES].append((assistant_reply, False))
