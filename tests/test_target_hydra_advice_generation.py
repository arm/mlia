# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for advice generation."""
from __future__ import annotations

from pathlib import Path

from mlia.backend.argo.config import ArgoConfig
from mlia.backend.argo.performance import ArgoPerformanceMetrics
from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.target.hydra.advice_generation import HydraAdviceProducer
from mlia.target.hydra.data_analysis import ModelPerformanceAnalysed


def test_hydra_advice_producer(tmpdir: str) -> None:
    """Test Hydra advice producer."""
    producer = HydraAdviceProducer()

    context = ExecutionContext(
        advice_category={AdviceCategory.PERFORMANCE},
        output_dir=tmpdir,
    )
    producer.set_context(context)

    metrics = ArgoPerformanceMetrics(
        backend_config=ArgoConfig(),
        metrics_file=Path("DOES_NOT_EXIST"),
        operator_performance_data=[],
    )
    producer.produce_advice(ModelPerformanceAnalysed(metrics))

    # Compatibility is not supported and should do nothing
    context.advice_category = {AdviceCategory.COMPATIBILITY}
    producer.set_context(context)
    producer.produce_advice(ModelPerformanceAnalysed(metrics))
