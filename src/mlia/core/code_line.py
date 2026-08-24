# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Derive source-line entities from retained code-stack entities."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import PurePosixPath, PureWindowsPath

import mlia.core.output_schema as schema
from mlia.core.entity_graph import (
    EntityGraphDeclaration,
    validate_entity_graph,
)


def _path_basename(value: str) -> str:
    """Return the final component from a Windows or POSIX path string."""
    windows_name = PureWindowsPath(value).name
    posix_name = PurePosixPath(value).name
    return windows_name if len(windows_name) < len(posix_name) else posix_name


def _safe_entity_id_part(value: str) -> str:
    """Return a readable entity-ID component without hierarchy separators."""
    sanitized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value).strip("_")
    return sanitized or "line"


def _code_line_id(file: str, line: int) -> str:
    """Return a stable result-local identity for one exact source line."""
    identity = json.dumps([file, line], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    name = f"{_path_basename(file)}:{line}"
    return f"{schema.ENTITY_KIND_CODE_LINE}/{_safe_entity_id_part(name)}-{digest}"


def _code_stack_location(entity: schema.Entity) -> tuple[str, int] | None:
    """Return a valid well-known code-stack source location."""
    if entity.kind != schema.ENTITY_KIND_CODE_STACK:
        return None
    file = entity.attributes.get("file")
    line = entity.attributes.get("line")
    if not isinstance(file, str) or not file or type(line) is not int or line <= 0:
        return None
    return file, line


def derive_code_line_entities(result: schema.Result) -> schema.Result:
    """Add one core-owned code-line parent for each retained stack location.

    Backend output must not contain code-line entities. Core derives them after
    configured entity collapse, groups retained code stacks by their exact
    schema-normalized ``file`` and ``line`` attributes, and materializes both
    sides of every new relationship.
    """
    existing_line_ids = sorted(
        entity.id
        for entity in result.entities
        if entity.kind == schema.ENTITY_KIND_CODE_LINE
    )
    if existing_line_ids:
        raise ValueError(
            "Backend output must not define core-derived code_line entities: "
            + ", ".join(existing_line_ids)
        )

    stacks_by_location: dict[tuple[str, int], list[str]] = {}
    for entity in result.entities:
        location = _code_stack_location(entity)
        if location is not None:
            stacks_by_location.setdefault(location, []).append(entity.id)
    if not stacks_by_location:
        return result

    graph = validate_entity_graph(
        [
            EntityGraphDeclaration(
                entity.id,
                tuple(entity.parent_ids),
                tuple(entity.child_ids),
                index,
            )
            for index, entity in enumerate(result.entities)
        ]
    )
    known_ids = {entity.id for entity in result.entities}
    line_id_by_stack: dict[str, str] = {}
    line_entities: list[schema.Entity] = []
    for (file, line), stack_ids in sorted(stacks_by_location.items()):
        line_id = _code_line_id(file, line)
        if line_id in known_ids:
            raise ValueError(
                f"Derived code_line entity id '{line_id}' conflicts with an "
                "existing entity id."
            )
        known_ids.add(line_id)
        sorted_stack_ids = sorted(stack_ids)
        line_entities.append(
            schema.Entity(
                id=line_id,
                kind=schema.ENTITY_KIND_CODE_LINE,
                name=f"{_path_basename(file)}:{line}",
                child_ids=sorted_stack_ids,
                attributes={"file": file, "line": line},
            )
        )
        line_id_by_stack.update(dict.fromkeys(sorted_stack_ids, line_id))

    entities = [
        (
            replace(
                entity,
                parent_ids=[
                    *sorted(graph.parents[entity.id]),
                    line_id_by_stack[entity.id],
                ],
            )
            if entity.id in line_id_by_stack
            else entity
        )
        for entity in result.entities
    ]
    return replace(result, entities=[*entities, *line_entities])
