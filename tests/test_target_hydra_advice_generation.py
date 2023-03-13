# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for advice generation."""
from __future__ import annotations

import pytest

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

    # Performance is not yet implemented
    with pytest.raises(NotImplementedError):
        producer.produce_advice(ModelPerformanceAnalysed({}))

    # Compatibility is not supported and should do nothing
    context.advice_category = {AdviceCategory.COMPATIBILITY}
    producer.set_context(context)
    producer.produce_advice(ModelPerformanceAnalysed({}))
