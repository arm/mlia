# Copyright 2021, Arm Ltd.
"""Module for advice generation."""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import List
from typing import Union

from mlia.core.common import DataItem
from mlia.core.events import SystemEvent
from mlia.core.mixins import ContextMixin


@dataclass
class Advice:
    """Base class for the advice."""

    msgs: List[str]


@dataclass
class AdviceEvent(SystemEvent):
    """Advice event.

    This event is published for every produced advice.

    :param advice: Advice instance
    """

    advice: Advice


class AdviceProducer(ABC):
    """Base class for the advice producer.

    Producer has two methods for advice generation:
      - produce_advice - used to generate advice based on provided
        data (analyzed data item from analyze stage)
      - get_advice - used for getting generated advice

    Advice producers that have predefined advice could skip
    implementation of produce_advice method.
    """

    @abstractmethod
    def produce_advice(self, data_item: DataItem) -> None:
        """Process data item and produce advice.

        :param data_item: piece of data that could be used
               for advice generation
        """

    @abstractmethod
    def get_advice(self) -> Union[Advice, List[Advice]]:
        """Get produced advice."""


class ContextAwareAdviceProducer(AdviceProducer, ContextMixin):
    """Context aware advice producer.

    This class makes easier access to the Context object. Context object could
    be automatically injected during workflow configuration.
    """


class FactBasedAdviceProducer(ContextAwareAdviceProducer):
    """Advice producer based on provided facts.

    This is an utility class that maintain list of generated Advice instances.
    """

    def __init__(self) -> None:
        """Init advice producer."""
        self.advice: List[Advice] = []

    def get_advice(self) -> Union[Advice, List[Advice]]:
        """Get produced advice."""
        return self.advice

    def add_advice(self, msgs: List[str]) -> None:
        """Add advice."""
        self.advice.append(Advice(msgs))
