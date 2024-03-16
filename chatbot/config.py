"""
LLM RAG Chatbot
"""
import json
from typing import Any


class Config:
    """
    The Config class represents a configuration object that maps the
    configuration of the application.

    It exposes a convenient getter to access the elements in
    `config.json`
    """

    def __init__(self):
        self._config = None

    @staticmethod
    def get(path: str) -> Any:
        parts = path.split(".")
        config = Config._get_config()
        for part in parts:
            config = config[part]
        return config

    @staticmethod
    def _get_config() -> Any:
        with open('config/config.json') as config:
            return json.load(config)
