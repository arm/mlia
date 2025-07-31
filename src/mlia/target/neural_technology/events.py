# SPDX-FileCopyrightText: Copyright 2023,2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Neural Technology MLIA module events."""
from dataclasses import dataclass
from pathlib import Path

from mlia.core.events import Event
from mlia.core.events import EventDispatcher
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration


@dataclass
class NeuralTechnologyAdvisorStartedEvent(Event):
    """Event with Neural Technology advisor parameters."""

    model: Path
    device: NeuralTechnologyConfiguration


class NeuralTechnologyAdvisorEventHandler(EventDispatcher):
    """Event handler for the Neural Technology inference advisor."""

    def on_neural_technology_advisor_started(
        self, event: NeuralTechnologyAdvisorStartedEvent
    ) -> None:
        """Handle NeuralTechnologyAdvisorStarted event."""
