# Copyright 2021, Arm Ltd.
"""CLI common module."""
import argparse
from pathlib import Path
from typing import Callable
from typing import List
from typing import Optional

from mlia.config import Context


class CommandInfo:
    """Command description."""

    def __init__(
        self,
        func: Callable,
        aliases: List[str],
        opt_groups: List[Callable[[argparse.ArgumentParser], None]],
        is_default: bool = False,
    ) -> None:
        """Init command."""
        self.func = func
        self.aliases = aliases
        self.opt_groups = opt_groups
        self.is_default = is_default

    @property
    def command_name(self) -> str:
        """Return command name."""
        return self.func.__name__

    @property
    def command_name_and_aliases(self) -> List[str]:
        """Return list of command name and aliases."""
        return [self.command_name, *self.aliases]

    @property
    def command_help(self) -> str:
        """Return help message for the command."""
        assert self.func.__doc__, "Command function does not have a docstring"
        func_help = self.func.__doc__.splitlines()[0].rstrip(".")

        if self.is_default:
            func_help = f"{func_help} [default]"

        return func_help


class ExecutionContext(Context):
    """Execution context."""

    def __init__(
        self,
        *,
        working_dir: Optional[str] = None,
        verbose: bool = False,
        logs_dir: str = "logs",
        models_dir: str = "models",
    ) -> None:
        """Init execution context.

        :param working_dir: path to the directory where application will store
               models, temporary files, logs, etc
        :param verbose: enable verbose output
        :param logs_dir: logs directory name inside working directory
        :param models_dir: models directory name inside working directory
        """
        self._working_dir_path = Path.cwd()
        if working_dir:
            self._working_dir_path = Path(working_dir)
            self._working_dir_path.mkdir(exist_ok=True)

        self.verbose = verbose
        self.logs_dir = logs_dir
        self.models_dir = models_dir

    def get_model_path(self, model_filename: str) -> Path:
        """Return path for the model."""
        models_dir_path = self._working_dir_path / self.models_dir
        models_dir_path.mkdir(exist_ok=True)

        return models_dir_path / model_filename

    @property
    def logs_path(self) -> Path:
        """Return path to the logs directory."""
        return self._working_dir_path / self.logs_dir

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"ExecutionContext: working_dir={self._working_dir_path}, "
            "verbose={self.verbose}"
        )
