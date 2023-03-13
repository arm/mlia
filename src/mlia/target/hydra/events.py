# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Hydra MLIA module events."""
from dataclasses import dataclass
from pathlib import Path

from mlia.core.events import Event
from mlia.core.events import EventDispatcher
from mlia.target.hydra.config import HydraConfiguration


@dataclass
class HydraAdvisorStartedEvent(Event):
    """Event with Hydra advisor parameters."""

    model: Path
    device: HydraConfiguration


class HydraAdvisorEventHandler(EventDispatcher):
    """Event handler for the Hydra inference advisor."""

    def on_hydra_advisor_started(self, event: HydraAdvisorStartedEvent) -> None:
        """Handle HydraAdvisorStarted event."""
