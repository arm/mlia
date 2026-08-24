# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Core postprocessing for authoritative standardized MLIA output."""

from __future__ import annotations

import copy
from typing import Any

import mlia.core.output_schema as schema
from mlia.backend.registry import ensure_backend_plugins_loaded, registry
from mlia.core.code_line import derive_code_line_entities
from mlia.core.entity_collapse import collapse_entities
from mlia.core.entity_graph import EntityGraphValidationError
from mlia.core.output_projection import project_entity_breakdowns
from mlia.core.output_validation import (
    SchemaValidationError,
    validate_standardized_output,
)
from mlia.core.settings import ApplicationSettings, CollapseRule


def _graph_error(error: EntityGraphValidationError) -> SchemaValidationError:
    details = "\n  - ".join(issue.message() for issue in error.issues)
    return SchemaValidationError(f"Entity graph validation failed:\n  - {details}")


def _collapse_rules(
    result: schema.Result, settings: ApplicationSettings
) -> tuple[CollapseRule, ...]:
    """Combine backend-owned defaults with explicit application rules."""
    ensure_backend_plugins_loaded()
    backend = registry.items.get(result.producer)
    defaults = backend.default_collapse_rules if backend is not None else ()
    return tuple(dict.fromkeys((*defaults, *settings.filtering.collapse)))


def postprocess_standardized_output(
    output: dict[str, Any], settings: ApplicationSettings
) -> dict[str, Any]:
    """Collapse, derive source lines, and project standardized results.

    Basic output and entity-graph validation runs before collapse so configuration
    cannot hide malformed backend output. Source lines are derived from the retained
    code stacks before projection. Every result passes through typed result
    validation, and the processed output is validated again before delivery.
    """
    validate_standardized_output(output, use_jsonschema=False)
    processed = copy.deepcopy(output)
    results = processed["results"]

    for index, raw_result in enumerate(results):
        try:
            parsed_result = schema.Result.from_dict(raw_result)
            result = parsed_result
            if result.entities:
                result = collapse_entities(result, _collapse_rules(result, settings))
                result = derive_code_line_entities(result)
                result = project_entity_breakdowns(result)
        except EntityGraphValidationError as err:
            raise _graph_error(err) from err
        except (KeyError, TypeError, ValueError) as err:
            raise SchemaValidationError(
                f"Unable to postprocess standardized result {index}: {err}"
            ) from err
        if result != parsed_result:
            results[index] = result.to_dict()

    validate_standardized_output(processed, use_jsonschema=False)
    return processed
