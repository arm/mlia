# Copyright 2021, Arm Ltd.
"""Inference advisor module."""
from abc import ABC
from abc import abstractmethod

from mlia.core.context import Context
from mlia.core.workflow import WorkflowExecutor


class InferenceAdvisor(ABC):
    """Base class for the inference advisor."""

    @abstractmethod
    def configure(self, context: Context) -> WorkflowExecutor:
        """Configure advisor execution."""

    def run(self, context: Context) -> None:
        """Run inference advisor."""
        executor = self.configure(context)
        executor.run()
