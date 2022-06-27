# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Context module.

This module contains functionality related to the Context.
Context is an object that describes advisor working environment
and requested behavior (advice categories, input configuration
parameters).
"""
import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import List
from typing import Mapping
from typing import Optional
from typing import Union

from mlia.core.common import AdviceCategory
from mlia.core.events import DefaultEventPublisher
from mlia.core.events import EventHandler
from mlia.core.events import EventPublisher
from mlia.core.helpers import ActionResolver
from mlia.core.helpers import APIActionResolver

logger = logging.getLogger(__name__)


class Context(ABC):
    """Abstract class for the execution context."""

    @abstractmethod
    def get_model_path(self, model_filename: str) -> Path:
        """Return path for the intermediate/optimized models.

        During workflow execution different parts of the advisor
        require creating intermediate files for models.

        This method allows to provide paths where those models
        could be saved.

        :param model_filename: filename of the model
        """

    @property
    @abstractmethod
    def event_publisher(self) -> EventPublisher:
        """Return event publisher."""

    @property
    @abstractmethod
    def event_handlers(self) -> Optional[List[EventHandler]]:
        """Return list of the event_handlers."""

    @property
    @abstractmethod
    def advice_category(self) -> AdviceCategory:
        """Return advice category."""

    @property
    @abstractmethod
    def config_parameters(self) -> Optional[Mapping[str, Any]]:
        """Return configuration parameters."""

    @property
    @abstractmethod
    def action_resolver(self) -> ActionResolver:
        """Return action resolver."""

    @abstractmethod
    def update(
        self,
        *,
        advice_category: AdviceCategory,
        event_handlers: List[EventHandler],
        config_parameters: Mapping[str, Any],
    ) -> None:
        """Update context parameters."""

    def category_enabled(self, category: AdviceCategory) -> bool:
        """Check if category enabled."""
        return category == self.advice_category

    def any_category_enabled(self, *categories: AdviceCategory) -> bool:
        """Return true if any category is enabled."""
        return self.advice_category in categories

    def register_event_handlers(self) -> None:
        """Register event handlers."""
        self.event_publisher.register_event_handlers(self.event_handlers)


class ExecutionContext(Context):
    """Execution context."""

    def __init__(
        self,
        *,
        advice_category: AdviceCategory = AdviceCategory.ALL,
        config_parameters: Optional[Mapping[str, Any]] = None,
        working_dir: Optional[Union[str, Path]] = None,
        event_handlers: Optional[List[EventHandler]] = None,
        event_publisher: Optional[EventPublisher] = None,
        verbose: bool = False,
        logs_dir: str = "logs",
        models_dir: str = "models",
        action_resolver: Optional[ActionResolver] = None,
    ) -> None:
        """Init execution context.

        :param advice_category: requested advice category
        :param config_parameters: dictionary like object with input parameters
        :param working_dir: path to the directory that will be used as a place
               to store temporary files, logs, models. If not provided then
               current working directory will be used instead
        :param event_handlers: optional list of event handlers
        :param event_publisher: optional event publisher instance. If not provided
               then default implementation of event publisher will be used
        :param verbose: enable verbose output
        :param logs_dir: name of the directory inside working directory where
               log files will be stored
        :param models_dir: name of the directory inside working directory where
               temporary models will be stored
        :param action_resolver: instance of the action resolver that could make
               advice actionable
        """
        self._advice_category = advice_category
        self._config_parameters = config_parameters

        self._working_dir_path = Path.cwd()
        if working_dir:
            self._working_dir_path = Path(working_dir)
            self._working_dir_path.mkdir(exist_ok=True)

        self._event_handlers = event_handlers
        self._event_publisher = event_publisher or DefaultEventPublisher()
        self.verbose = verbose
        self.logs_dir = logs_dir
        self.models_dir = models_dir
        self._action_resolver = action_resolver or APIActionResolver()

    @property
    def advice_category(self) -> AdviceCategory:
        """Return advice category."""
        return self._advice_category

    @advice_category.setter
    def advice_category(self, advice_category: AdviceCategory) -> None:
        """Setter for the advice category."""
        self._advice_category = advice_category

    @property
    def config_parameters(self) -> Optional[Mapping[str, Any]]:
        """Return configuration parameters."""
        return self._config_parameters

    @config_parameters.setter
    def config_parameters(self, config_parameters: Optional[Mapping[str, Any]]) -> None:
        """Setter for the configuration parameters."""
        self._config_parameters = config_parameters

    @property
    def event_handlers(self) -> Optional[List[EventHandler]]:
        """Return list of the event handlers."""
        return self._event_handlers

    @event_handlers.setter
    def event_handlers(self, event_handlers: List[EventHandler]) -> None:
        """Setter for the event handlers."""
        self._event_handlers = event_handlers

    @property
    def event_publisher(self) -> EventPublisher:
        """Return event publisher."""
        return self._event_publisher

    @property
    def action_resolver(self) -> ActionResolver:
        """Return action resolver."""
        return self._action_resolver

    def get_model_path(self, model_filename: str) -> Path:
        """Return path for the model."""
        models_dir_path = self._working_dir_path / self.models_dir
        models_dir_path.mkdir(exist_ok=True)

        return models_dir_path / model_filename

    @property
    def logs_path(self) -> Path:
        """Return path to the logs directory."""
        return self._working_dir_path / self.logs_dir

    def update(
        self,
        *,
        advice_category: AdviceCategory,
        event_handlers: List[EventHandler],
        config_parameters: Mapping[str, Any],
    ) -> None:
        """Update context parameters."""
        self._advice_category = advice_category
        self._event_handlers = event_handlers
        self._config_parameters = config_parameters

    def __str__(self) -> str:
        """Return string representation."""
        category = (
            "<not set>" if self.advice_category is None else self.advice_category.name
        )

        return (
            f"ExecutionContext: working_dir={self._working_dir_path}, "
            f"advice_category={category}, "
            f"config_parameters={self.config_parameters}, "
            f"verbose={self.verbose}"
        )
