# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Argo backend configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ArgoConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for the Argo compiler."""

    accelerator_config: str = "hydra"
    system_config: str | None = None
    npu_clock_mhz: int | None = None
    num_npu_cores: int | None = None
    num_active_npu_cores: int | None = None
    num_cpu_cores: int | None = None
    dram_bandwidth_gb_per_sec: int | None = None
    gsb_size_kb: int | None = None
    l1_cache_size_kb: int | None = None
    l2_cache_size_kb: int | None = None
    system_sram_size_kb: int | None = None
