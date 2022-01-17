# Copyright 2021, Arm Ltd.
"""Module for custom type hints."""
from pathlib import Path
from typing import Literal
from typing import TextIO
from typing import Union


FileLike = TextIO
PathOrFileLike = Union[str, Path, FileLike]
OutputFormat = Literal["plain_text", "csv", "json"]
