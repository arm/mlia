# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for testing AIET main.py."""
from typing import Any
from unittest.mock import MagicMock

from aiet import main


def test_main(monkeypatch: Any) -> None:
    """Test main entry point function."""
    with monkeypatch.context() as mock_context:
        mock = MagicMock()
        mock_context.setattr(main, "cli", mock)
        main.main()
        mock.assert_called_once()
