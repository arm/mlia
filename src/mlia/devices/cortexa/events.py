# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A MLIA module events."""
from dataclasses import dataclass
from pathlib import Path

from mlia.core.events import Event
from mlia.core.events import EventDispatcher
from mlia.devices.cortexa.config import CortexAConfiguration


@dataclass
class CortexAAdvisorStartedEvent(Event):
    """Event with Cortex-A advisor parameters."""

    model: Path
    device: CortexAConfiguration


class CortexAAdvisorEventHandler(EventDispatcher):
    """Event handler for the Cortex-A inference advisor."""

    def on_cortex_a_advisor_started(self, event: CortexAAdvisorStartedEvent) -> None:
        """Handle CortexAAdvisorStarted event."""
