# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for the installation of Argo via Docker."""
from __future__ import annotations

from pathlib import Path

from mlia.backend.argo.config import ArgoConfig


def estimate_performance(model_path: Path, cfg: ArgoConfig) -> Path:
    """Return the path to the Argo stats file."""
    raise NotImplementedError(f"TODO: Run Argo here!\n{model_path}\n{cfg}")
    # argo_output_dir = Path("argo_output")
    # argo_stats_file = f"{model_path.stem}_chrome_trace.json.json"
    # return argo_output_dir / argo_stats_file
