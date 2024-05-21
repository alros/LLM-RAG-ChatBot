"""
LLM RAG Chatbot
"""
from json import JSONDecodeError
from typing import Tuple
import streamlit as st
from streamlit_chat import message
from chatbot.config import Config
from steps import Step


class Page:
    """
    The Page class represents the single page Chatbot application.
    """

    # Some constants to identify elements in the page's construction
    SESSION_MESSAGES = 'messages'
    USER_INPUT = 'user_input'
    SPINNER_THINKING = 'thinking_spinner'
    FINAL_DIAGNOSIS = 'final_diagnosis'
    DISCUSSION = 'discussion'
    PATIENT_SUMMARY = 'patient_summary'
    SKIP_MESSAGES = 'skip_messages'

    def __init__(self,
                 step_chat: Step,
                 step_summary: Step,
                 step_diagnosis: Step,
                 step_final_diagnosis: Step,
                 step_discussion: Step):
        """
        Initialises the Page.

        :param step_chat: instance of the Step class for the chat step.
        :param step_summary: instance of the Step class for the summary step.
        :param step_diagnosis: instance of the Step class for the diagnosis step.
        :param step_final_diagnosis: instance of the Step class for the final diagnosis step.
        :param step_discussion: instance of the Step to discuss with the patient after the diagnosis.
        """

        self._step_chat = step_chat
        self._step_summary = step_summary
        self._step_diagnosis = step_diagnosis
        self._step_final_diagnosis = step_final_diagnosis
        self._step_discussion = step_discussion

        # the following code configures the page with streamlit.
        st.set_page_config(page_title=Config.get('page.title'))
        if Page.SESSION_MESSAGES not in st.session_state:
            # init some variables
            st.session_state[Page.SESSION_MESSAGES] = []
            st.session_state[Page.SPINNER_THINKING] = st.empty()
            st.session_state[Page.SKIP_MESSAGES] = 0

        st.header(Config.get('page.header'))
        st.subheader(Config.get('page.subHeader'))
        st.warning(Config.get('page.warning'), icon="⚠️")

        if Page.DISCUSSION in st.session_state:
            self._discuss()
        elif Page.FINAL_DIAGNOSIS in st.session_state and st.session_state[Page.FINAL_DIAGNOSIS] is not None:
            self._close_diagnosis_session(st.session_state[Page.FINAL_DIAGNOSIS])
        else:
            self._continue_the_chat()

    def _continue_the_chat(self) -> None:
        """
        The chat did not reach the end yet, and there will be another loop.

        :return: None
        """

        # Generate the next question
        next_question = self._get_next_question()
        # store it among the messages
        st.session_state[Page.SESSION_MESSAGES].append({
            'q': next_question
        })
        # for an update
        self._update_the_page()
        # set the handle to react on the user's input.
        st.text_input(Config.get('page.userInputSuggestion'), key=Page.USER_INPUT, on_change=self._process_input)

    def _close_diagnosis_session(self, final_diagnosis: str) -> None:
        """
        There was a final diagnosis, and the session is closed.

        :param final_diagnosis: the final diagnosis.
        :return: None
        """

        st.session_state[Page.SESSION_MESSAGES].append({
            'q': final_diagnosis
        })
        questions, _ = self._get_chat()
        st.session_state[Page.SKIP_MESSAGES] = questions
        st.session_state[Page.SESSION_MESSAGES].append({
            'q': Config.get('chat.startDiscussion'),
        })
        self._update_the_page()
        st.text_input(Config.get('page.userInputSuggestion'), key=Page.USER_INPUT,
                      on_change=self._process_input_discussion)
        st.session_state[Page.DISCUSSION] = True

    def _discuss(self) -> None:
        with st.session_state[Page.SPINNER_THINKING], st.spinner(Config.get('page.spinnerText')):
            next_answer = self._get_next_answer()
        st.session_state[Page.SESSION_MESSAGES].append({
            'q': next_answer,
        })
        self._update_the_page()
        st.text_input(Config.get('page.userInputSuggestion'), key=Page.USER_INPUT,
                      on_change=self._process_input_discussion)

    def _update_the_page(self) -> None:
        """
        Copies the messages in the session into the streamlit data structure

        :return: None
        """
        for i, msg in enumerate(st.session_state[Page.SESSION_MESSAGES]):
            # messages is a structure in streamlit chat
            # tuples are in q: ... / a: ... format (the answer is optional)
            message(msg['q'], is_user=False, key=str(i) + 'q')
            if 'a' in msg:
                message(msg['a'], is_user=True, key=str(i) + 'a')
        # reset the spinner
        st.session_state[Page.SPINNER_THINKING] = st.empty()

    def _get_next_question(self) -> str:
        """
        Generates the next question.

        :return: the next question.
        """
        _, query = self._get_chat()
        return self._step_chat.query(query)

    def _get_next_answer(self) -> str:
        query = self._get_last_message()
        return self._step_discussion.query(query, summary=st.session_state[Page.PATIENT_SUMMARY])

    def _get_chat(self) -> Tuple[int, str]:
        """
        Returns the number of questions and the chat.
        The chat is in the format:
            You: question.
            Patient: reply.
        :return: (number of questions, chat).
        """
        messages = st.session_state[Page.SESSION_MESSAGES]
        chat = ''
        for idx, qa in enumerate(messages):
            if idx >= st.session_state[Page.SKIP_MESSAGES]:
                chat = chat + '\n' if chat else ''
                chat = f'{chat}You: "{qa["q"]}"'
                chat = f'{chat}\nPatient: "{qa["a"]}"' if 'a' in qa else chat
        return len(messages), chat

    def _get_last_message(self) -> str:
        messages = st.session_state[Page.SESSION_MESSAGES]
        return messages[len(messages)-1]['a']

    def _process_input(self) -> None:
        """
        Handles the submission of input from the user. This is called by streamlit.
        :return: None
        """
        if st.session_state[Page.USER_INPUT] and len(st.session_state[Page.USER_INPUT].strip()) > 0:
            self._process_user_text(st.session_state[Page.USER_INPUT].strip())

    def _process_user_text(self, user_text: str) -> None:
        """
        This method processes what the user wrote.
        This is the main loop of the application.

        :param user_text: input from the user.
        :return: None
        """
        st.session_state[Page.SESSION_MESSAGES][len(st.session_state[Page.SESSION_MESSAGES]) - 1]['a'] = user_text
        with st.session_state[Page.SPINNER_THINKING], st.spinner(Config.get('page.spinnerText')):
            # get the chat
            number_of_questions, chat = self._get_chat()
            # make a summary to describe the patient
            summary = self._step_summary.query(chat)
            st.session_state[Page.PATIENT_SUMMARY] = summary
            done = False
            attempts = 0
            # this while-loop is a workaround: sometimes the LLM does not respond in
            # a json format and the parsing fails. This is a mechanism to retry.
            while not done and attempts < 3:
                try:
                    # diagnose the patient
                    diagnostic = self._step_diagnosis.query(summary)
                    # if conditions are met, formulate a diagnosis.
                    # This is None if there were not enough questions, or if the
                    # score is not high enough. It can also be a negative diagnosis
                    # when the questions are beyond a certain amount.
                    final_diagnosis = self._step_final_diagnosis.query(diagnostic,
                                                                       number_of_questions=number_of_questions)
                    st.session_state[Page.FINAL_DIAGNOSIS] = final_diagnosis
                    done = True
                except JSONDecodeError:
                    # deal with non-json responses in the diagnostic step.
                    attempts = attempts + 1
                    st.session_state[Page.FINAL_DIAGNOSIS] = None
            # reset the input box
            st.session_state[Page.USER_INPUT] = ''

    def _process_input_discussion(self) -> None:
        if st.session_state[Page.USER_INPUT] and len(st.session_state[Page.USER_INPUT].strip()) > 0:
            user_text = st.session_state[Page.USER_INPUT].strip()
            st.session_state[Page.SESSION_MESSAGES][len(st.session_state[Page.SESSION_MESSAGES]) - 1]['a'] = user_text
            st.session_state[Page.USER_INPUT] = ''
