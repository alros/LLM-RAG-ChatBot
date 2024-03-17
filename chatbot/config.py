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

    @staticmethod
    def get(path: str) -> Any:
        """
        Return a value from the configuration.

        The path is a dot-separated string that specifies the location of the
        value in the configuration. E.g. `foo.bar` with `{"foo":{"bar":"value"}}`
        returns `value`.

        :param path: json path to the value.
        :return: the value from `config.yaml`.
        """
        parts = path.split(".")
        config = Config._get_config()
        for part in parts:
            config = config[part]
        return config

    @staticmethod
    def _get_config() -> Any:
        """
        Internal utility to open `config.json`

        :return: a dict containing nested dicts.
        """
        with open('config/config.json', encoding="utf8") as config:
            return json.load(config)
