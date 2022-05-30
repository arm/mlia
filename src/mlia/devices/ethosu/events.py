# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U MLIA module events."""
from dataclasses import dataclass
from pathlib import Path

from mlia.core.events import Event
from mlia.core.events import EventDispatcher
from mlia.devices.ethosu.config import EthosUConfiguration


@dataclass
class EthosUAdvisorStartedEvent(Event):
    """Event with Ethos-U advisor parameters."""

    model: Path
    device: EthosUConfiguration


class EthosUAdvisorEventHandler(EventDispatcher):
    """Event handler for the Ethos-U inference advisor."""

    def on_ethos_u_advisor_started(self, event: EthosUAdvisorStartedEvent) -> None:
        """Handle EthosUAdvisorStarted event."""
