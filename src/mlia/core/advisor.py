# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Inference advisor module."""
from abc import abstractmethod

from mlia.core.common import NamedEntity
from mlia.core.context import Context
from mlia.core.workflow import WorkflowExecutor


class InferenceAdvisor(NamedEntity):
    """Base class for inference advisors."""

    @abstractmethod
    def configure(self, context: Context) -> WorkflowExecutor:
        """Configure advisor execution."""

    def run(self, context: Context) -> None:
        """Run inference advisor."""
        executor = self.configure(context)
        executor.run()
