# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA advisor events."""
from dataclasses import dataclass
from pathlib import Path

from mlia.core.events import Event
from mlia.core.events import EventDispatcher
from mlia.target.tosa.config import TOSAConfiguration


@dataclass
class TOSAAdvisorStartedEvent(Event):
    """Event with TOSA advisor parameters."""

    model: Path
    device: TOSAConfiguration


class TOSAAdvisorEventHandler(EventDispatcher):
    """Event handler for the TOSA inference advisor."""

    def on_tosa_advisor_started(self, event: TOSAAdvisorStartedEvent) -> None:
        """Handle TOSAAdvisorStartedEvent event."""
