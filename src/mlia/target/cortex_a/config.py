# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A configuration."""
from __future__ import annotations

from typing import Any

from mlia.backend.armnn_tflite_delegate.compat import ARMNN_TFLITE_DELEGATE
from mlia.target.config import TargetProfile


class CortexAConfiguration(TargetProfile):
    """Cortex-A configuration."""

    def __init__(self, **kwargs: Any) -> None:
        """Init Cortex-A target configuration."""
        target = kwargs["target"]
        super().__init__(target)

        self.backend_config = kwargs.get("backend")
        self.armnn_tflite_delegate_version = kwargs["backend"]["armnn-tflite-delegate"][
            "version"
        ]

    def verify(self) -> None:
        """Check the parameters."""
        super().verify()
        if self.target != "cortex-a":
            raise ValueError(f"Wrong target {self.target} for Cortex-A configuration.")

        if not self.armnn_tflite_delegate_version:
            raise ValueError("No version for ArmNN TensorFlow Lite delegate specified.")
        if self.armnn_tflite_delegate_version not in ARMNN_TFLITE_DELEGATE["ops"]:
            raise ValueError(
                f"Version '{self.armnn_tflite_delegate_version}' of "
                f"backend {ARMNN_TFLITE_DELEGATE['backend']} is not supported. "
                f"Choose from: {', '.join(ARMNN_TFLITE_DELEGATE['ops'])}"
            )
