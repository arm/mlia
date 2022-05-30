# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for testing helpers.py."""
import logging
from typing import Any
from typing import List
from unittest.mock import call
from unittest.mock import MagicMock

import pytest

from aiet.utils.helpers import set_verbosity


@pytest.mark.parametrize(
    "verbosity,expected_calls",
    [(0, []), (1, [call(logging.INFO)]), (2, [call(logging.DEBUG)])],
)
def test_set_verbosity(
    verbosity: int, expected_calls: List[Any], monkeypatch: Any
) -> None:
    """Test set_verbosity() with different verbsosity levels."""
    with monkeypatch.context() as mock_context:
        logging_mock = MagicMock()
        mock_context.setattr(logging.getLogger(), "setLevel", logging_mock)
        set_verbosity(None, None, verbosity)
        logging_mock.assert_has_calls(expected_calls)
