from abc import ABC

from config import Config


class Prompts(ABC):
    def __init__(self, config_path: str):
        config = Config.get(config_path)
        self._system_prompt = '\n'.join(config['system'])
        self._user_prompt = '\n'.join(config['user'])

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def user_prompt(self) -> str:
        return self._user_prompt


class ChatPrompts(Prompts):
    def __init__(self):
        super().__init__('prompts.chat')


class SummaryPrompts(Prompts):
    def __init__(self):
        super().__init__('prompts.summary')


class DiagnosisPrompts(Prompts):
    def __init__(self):
        super().__init__('prompts.diagnosis')
