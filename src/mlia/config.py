# Copyright 2021, Arm Ltd.
"""Configuration module."""
# pylint: disable=too-few-public-methods,too-many-instance-attributes
# pylint: disable=too-many-arguments
import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class Context(ABC):
    """Abstract class for the execution context."""

    @abstractmethod
    def get_model_path(self, model_filename: str) -> Path:
        """Return path for the model."""
