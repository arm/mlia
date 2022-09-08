# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for custom type hints."""
from pathlib import Path
from typing import Literal
from typing import TextIO
from typing import Union


FileLike = TextIO
PathOrFileLike = Union[str, Path, FileLike]
OutputFormat = Literal["plain_text", "csv", "json"]
