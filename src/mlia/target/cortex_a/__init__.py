# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A target module."""
from mlia.target.registry import registry
from mlia.target.registry import TargetInfo

registry.register("cortex-a", TargetInfo(["ArmNNTFLiteDelegate"]))
