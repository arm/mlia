# Copyright 2021, Arm Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Module for running tests.

This file contains tests which will be execute by pytest.
Please refer to official pytest documentation.

https://docs.pytest.org/en/latest/contents.html
"""
from typing import Any

import pytest
from mlia.cli.main import main


def test_option_version(capfd: Any) -> None:
    """Test --version."""
    with pytest.raises(SystemExit) as ex:
        main(["--version"])

    assert ex.type == SystemExit
    assert ex.value.code == 0

    stdout, stderr = capfd.readouterr()
    assert len(stdout.splitlines()) == 1
    assert stderr == ""
