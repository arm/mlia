# SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""NGP operator compatibility module."""
from __future__ import annotations

import logging
import re
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from mlia.backend.repo import get_backend_repository
from mlia.backend.vulkan_model_converter.conversion import VulkanModelConverterBase
from mlia.nn.tensorflow.tflite_graph import operator_names_to_types

logger = logging.getLogger(__name__)


class VMCCompatibilityLogReader:
    """Read log from VMC and extract low-level NGP compatibility information."""

    _lowered_ops: dict[str, str]
    _lowering_errors: dict[str, str]

    def __init__(self) -> None:
        """Initialize cumulative fields and patterns."""
        self._lowered_ops = {}
        self._lowering_errors = {}
        self._loc_pattern = re.compile(r'loc\((?:"(\S+)"|fused\[(.*?)\])\)')
        self._success_pattern = re.compile(r"^Successfully lowered: (\S*)\s+(at)? (.*)")
        self._error_pattern = re.compile(r"^\S+: error: (.*?): (.*?)$")

    def __call__(self, line: str) -> None:
        """Redirect output to the logger."""
        if match := self._success_pattern.match(line):
            lowered_op, filling, rest = match.group(1, 2, 3)
            if filling != "at":
                raise RuntimeError("Unrecognized log line: '{line}'")

            loc_string = self.parse_loc(rest)
            self._lowered_ops[loc_string] = lowered_op
        elif match := self._error_pattern.match(line):
            loc_part, error = match.group(1, 2)
            loc_string = self.parse_loc(loc_part)
            self._lowering_errors[loc_string] = error

    @property
    def lowered_ops(self) -> dict[str, str]:
        """Return a mapping of source(TF) to lowered(TOSA) operators."""
        return self._lowered_ops

    @property
    def lowering_errors(self) -> dict[str, str]:
        """Return a mapping of source(TF) to error messages of affected operators."""
        return self._lowering_errors

    def parse_loc(self, line: str) -> str:
        """Parse loc() expression in log strings."""
        if loc_match := self._loc_pattern.match(line):
            loc_string, fused_list = loc_match.group(1, 2)
            if loc_string:
                return loc_string
            if fused_list:
                locs = fused_list.split(", ")
                loc_strings = [loc.strip('"') for loc in locs]
                return loc_strings[0]
        raise RuntimeError(f"Can't find a valid location string in {line}")


class VMCCompatbilityChecker(VulkanModelConverterBase):
    """Run the Vulkan Model Converter to check for NGP compatibility."""

    def __init__(self, converter_path: Path) -> None:
        """Set up compatilibity checking for Vulkan Model Converter."""
        super().__init__(converter_path)
        self._compatibility_log_reader = VMCCompatibilityLogReader()
        self.output_consumers.append(self._compatibility_log_reader)

    def _extra_back_end_arguments(self) -> list[str]:
        """Return any extra arguments to be used with the VMC back-end."""
        return ["--experimental-analysis"]

    def _extra_front_end_arguments(self) -> list[str]:
        """Return any extra arguments to be used with the VMC front-end."""
        return self._extra_back_end_arguments() + ["--emit-byte-code"]

    @property
    def compatibility_log_reader(self) -> VMCCompatibilityLogReader:
        """Return the log reader which holds to operator-level details."""
        return self._compatibility_log_reader


@dataclass
class NGPOperatorCompatibilityInfo:
    """Describes a particular operator's compatibility with NGP."""

    location: str
    compat_level: str | None = None
    type: str | None = None
    tosa_op: str | None = None
    error: str | None = None
    placement: str | None = None


class NGPModelCompatibilityInfo:
    """Contains information about a model's compatibility with NGP."""

    _layer_map: dict[str, NGPOperatorCompatibilityInfo]

    def __init__(self, location_to_type: dict[str, str] | None = None) -> None:
        """Initialize the database."""
        self._layer_map = {}
        self._location_to_type = location_to_type or {}

    def _find_or_create_record(self, location: str) -> NGPOperatorCompatibilityInfo:
        """Get a record for a particular model location, create if necessary."""
        record: NGPOperatorCompatibilityInfo | None = self._layer_map.get(location)
        if record is None:
            record = NGPOperatorCompatibilityInfo(location)
            record.type = self._location_to_type.get(location)
            self._layer_map[location] = record
        return record

    def add_lowered_to_tosa(self, location: str, tosa_op: str) -> None:
        """Add an op to the database, that was reported to be lowered to TOSA."""
        record = self._find_or_create_record(location)
        record.tosa_op = tosa_op
        is_shader_op = tosa_op == "tosa.custom"
        if is_shader_op:
            record.compat_level = "Shader"
            record.placement = "EE"
        else:
            record.compat_level = "TOSA"
            record.placement = "NE"

    def add_lowering_error(self, location: str, error: str) -> None:
        """Add an op to the database, which can't be lowered due to some error."""
        record = self._find_or_create_record(location)
        record.error = error
        record.compat_level = "Non-NGP"

    @property
    def layer_map(self) -> dict[str, NGPOperatorCompatibilityInfo]:
        """Returns the underlying compatibilty records mapped to locations strings."""
        return self._layer_map

    def get_records(self) -> list[NGPOperatorCompatibilityInfo]:
        """Return an ordered list of records."""
        return [self.layer_map[loc] for loc in sorted(self.layer_map.keys())]

    def dump(self) -> list[dict]:
        """Dump info into a list of strings, for testing purposes."""

        def filter_no_values(k2v: dict) -> dict:
            return {k: v for k, v in sorted(k2v.items()) if v}

        return [filter_no_values(asdict(record)) for record in self.get_records()]


class NGPCompatibilityChecker:
    """Checker operator compability for NGP targets."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize the checker."""
        self.output_dir = output_dir

    def check_compatibility(self, tflite_model_path: Path) -> NGPModelCompatibilityInfo:
        """Run compabitlity check using Vulkan Model Converter."""
        backend_repo = get_backend_repository()
        vmc_path, _ = backend_repo.get_backend_settings("vulkan-model-converter")
        output_dir = self.output_dir / "vulkan-model-converter"
        output_dir.mkdir()

        vmc = VMCCompatbilityChecker(vmc_path)
        vmc(tflite_model_path, output_dir)
        reader: VMCCompatibilityLogReader = vmc.compatibility_log_reader

        comp_info = NGPModelCompatibilityInfo(
            operator_names_to_types(tflite_model_path)
        )

        for lowered_op, tosa_op in reader.lowered_ops.items():
            comp_info.add_lowered_to_tosa(lowered_op, tosa_op)

        for location, error in reader.lowering_errors.items():
            comp_info.add_lowering_error(location, error)

        return comp_info
