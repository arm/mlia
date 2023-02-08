# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA advisor events."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mlia.core.events import Event
from mlia.core.events import EventDispatcher
from mlia.target.tosa.config import TOSAConfiguration
from mlia.target.tosa.reporters import MetadataDisplay


@dataclass
class TOSAAdvisorStartedEvent(Event):
    """Event with TOSA advisor parameters."""

    model: Path
    target: TOSAConfiguration
    tosa_metadata: MetadataDisplay | None


class TOSAAdvisorEventHandler(EventDispatcher):
    """Event handler for the TOSA inference advisor."""

    def on_tosa_advisor_started(self, event: TOSAAdvisorStartedEvent) -> None:
        """Handle TOSAAdvisorStartedEvent event."""
