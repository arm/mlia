# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ethos-U MLIA module."""
from pathlib import Path

import pytest

from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.target.ethos_u.advisor import configure_and_get_ethosu_advisor
from mlia.target.ethos_u.advisor import EthosUInferenceAdvisor


def test_advisor_metadata() -> None:
    """Test advisor metadata."""
    assert EthosUInferenceAdvisor.name() == "ethos_u_inference_advisor"


def test_unsupported_advice_categories(tmp_path: Path, test_tflite_model: Path) -> None:
    """Test that advisor should throw an exception for unsupported categories."""
    with pytest.raises(
        Exception, match="Optimizations are not supported for TensorFlow Lite files."
    ):
        ctx = ExecutionContext(
            output_dir=tmp_path, advice_category={AdviceCategory.OPTIMIZATION}
        )

        advisor = configure_and_get_ethosu_advisor(
            ctx, "ethos-u55-256", str(test_tflite_model)
        )
        advisor.configure(ctx)
