# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test configuration for the end-to-end tests."""
from typing import cast

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command line options to control the e2e test behavior."""
    parser.addoption(
        "--no-skip",
        action="store_true",
        help="If set, forces tests to run regardless of the availability of "
        "MLIA backends required for the test. If not set, tests will be "
        "skipped if the required backend is not available.",
    )


@pytest.fixture(scope="session", name="no_skip")
def fixture_no_skip(request: pytest.FixtureRequest) -> bool:
    """Fixture for easy access to the '--no-skip' parameter."""
    return cast(bool, request.config.getoption("--no-skip", default=True))
