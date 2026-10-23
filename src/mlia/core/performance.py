# SPDX-FileCopyrightText: Copyright 2022, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for performance estimation."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import mlia
from mlia.core import output_schema as schema
from mlia.utils.filesystem import sha256

M = TypeVar("M")  # model type
P = TypeVar("P")  # performance metrics


@dataclass
class PerformanceObservations:
    """Target-neutral measured entities, relationships and metrics."""

    metrics: list[schema.Metric] = field(default_factory=list)
    entities: list[schema.Entity] = field(default_factory=list)
    breakdowns: list[schema.Breakdown] = field(default_factory=list)
    entity_kinds: list[schema.EntityKind] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class MeasuredPerformanceMetrics:
    """Measured results that require no estimator or hardware statistics."""

    measured_observations: list[PerformanceObservations]
    backend_configuration: dict[str, Any] = field(default_factory=dict)
    runtime_configuration: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)
    model_format: str | None = None
    mode: schema.ModeType = schema.ModeType.MEASURED

    def to_standardized_output(
        self,
        model_path: Path,
        *,
        target: schema.Target,
        backend: schema.Backend,
        cli_arguments: list[str] | None = None,
    ) -> dict[str, Any]:
        """Serialize observations using target metadata supplied by the caller.

        Measured backend configuration overrides matching caller-supplied keys.
        """
        backend = replace(
            backend,
            configuration={**backend.configuration, **self.backend_configuration},
        )
        return schema.StandardizedOutput(
            schema_version=schema.SCHEMA_VERSION,
            run_id=schema.StandardizedOutput.create_run_id(),
            timestamp=schema.StandardizedOutput.create_timestamp(),
            tool=schema.Tool(name="mlia", version=mlia.__version__),
            target=target,
            model=schema.Model(
                name=model_path.name,
                format=self.model_format or model_path.suffix.lstrip("."),
                hash=sha256(model_path),
                size_bytes=model_path.stat().st_size,
            ),
            context=schema.Context(
                cli_arguments=cli_arguments or [],
                runtime_configuration=self.runtime_configuration,
            ),
            backends=[backend],
            results=[
                schema.Result(
                    kind=schema.ResultKind.PERFORMANCE,
                    status=schema.ResultStatus.OK,
                    producer=backend.id,
                    mode=self.mode,
                    metrics=item.metrics,
                    entities=item.entities,
                    breakdowns=item.breakdowns,
                    entity_kinds=item.entity_kinds,
                    warnings=item.warnings,
                )
                for item in self.measured_observations
            ],
            extensions=self.extensions,
        ).to_dict()


class PerformanceEstimator(Generic[M, P]):
    """Base class for the performance estimation."""

    @abstractmethod
    def estimate(self, model: M) -> P:
        """Estimate performance."""


def estimate_performance(
    original_model: M,
    estimator: PerformanceEstimator[M, P],
    model_transformations: list[Callable[[M], M]],
) -> list[P]:
    """Estimate performance impact.

    This function estimates performance impact on model performance after
    applying provided transformations/optimizations.

    :param original_model: object that represents a model, could be
           instance of the model or path to the model. This depends on
           provided performance estimator.
    :param estimator: performance estimator
    :param model_transformations: list of the callables each of those
           returns object that represents optimized model
    """
    original_metrics = estimator.estimate(original_model)

    optimized_metrics = [
        estimator.estimate(transform(original_model))
        for transform in model_transformations
    ]

    return [original_metrics, *optimized_metrics]
