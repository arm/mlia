# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U target module."""
from mlia.target.registry import registry
from mlia.target.registry import TargetInfo

registry.register("Ethos-U55", TargetInfo(["Vela", "Corstone-300", "Corstone-310"]))
registry.register("Ethos-U65", TargetInfo(["Vela", "Corstone-300", "Corstone-310"]))
