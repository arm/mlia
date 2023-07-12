# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.core.extract."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Iterable

import pytest

from mlia.nn.rewrite.core.extract import ExtractPaths
from mlia.nn.rewrite.core.graph_edit.record import DEQUANT_SUFFIX


@pytest.mark.parametrize("dir_path", ("/dev/null", Path("/dev/null")))
@pytest.mark.parametrize("model_is_quantized", (False, True))
@pytest.mark.parametrize(
    ("obj", "func_names", "suffix"),
    (
        (ExtractPaths.tflite, ("start", "replace", "end"), ".tflite"),
        (ExtractPaths.tfrec, ("input", "output", "end"), ".tfrec"),
    ),
)
def test_extract_paths(
    dir_path: str | Path,
    model_is_quantized: bool,
    obj: Any,
    func_names: Iterable[str],
    suffix: str,
) -> None:
    """Test class ExtractPaths."""
    for func_name in func_names:
        func = getattr(obj, func_name)
        path = func(dir_path, model_is_quantized)
        assert path == Path(dir_path, path.relative_to(dir_path))
        assert path.suffix == suffix
        assert not model_is_quantized or path.stem.endswith(DEQUANT_SUFFIX)
