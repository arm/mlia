# SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Module to track stripe-level statistics to TFLite granularity."""
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict
from typing import Union

from mlia.backend.ngp_graph_compiler.output_parsing import DebugDatabaseContentsType
from mlia.backend.ngp_graph_compiler.output_parsing import (
    PerformanceDatabaseContentsType,
)


@dataclass
class NGPOperatorPerformanceStats:
    """Defines the performance stats for one operator."""

    op_id: list
    op_cycles: int
    total_cycles: int
    memory: dict
    utilization: list
    operators: list

    def to_dict(self) -> dict:
        """Convert one object to dictionary."""
        return {
            "stripe_ids": self.op_id,
            "op_cycles": self.op_cycles,
            "total_cycles": self.total_cycles,
            "memory": self.memory,
            "utilization": self.utilization,
            "operators": self.operators,
        }

    def sanitize_memory_fields(self) -> None:
        """Remove Undefined and Internal memory fields as they are meaningless."""
        del self.memory["Undefined"]
        del self.memory["Internal"]

    def sanitize_utilization_fields(self) -> None:
        """Aggregrate and sanitize utilization fields."""
        utilization_dict = defaultdict(list)
        for util in self.utilization:
            try:
                section_name = util["sectionName"]
                hw_util = util["hwUtil"]
                utilization_dict[section_name].append(float(hw_util))
            except KeyError as exc:
                raise KeyError(
                    "sectionName or hwUtil missing from the utilization statistics."
                ) from exc

        self.utilization = []

        for util_k, util_v in utilization_dict.items():
            util = {
                "sectionName": util_k,
                "hwUtil": str(round(sum(util_v) / len(util_v), 3)),
            }
            self.utilization.append(util)

    def merge(self, op_perf_stats: "NGPOperatorPerformanceStats") -> None:
        """Merge statistics belonging to different stripes."""
        if self.operators != op_perf_stats.operators:
            raise ValueError("The same chain should map to the same location strings!")

        self.op_id.extend(op_perf_stats.op_id)
        self.op_cycles += op_perf_stats.op_cycles
        self.total_cycles += op_perf_stats.total_cycles

        for key, value in self.memory.items():
            try:
                to_merge = op_perf_stats.memory[key]
                merged_mem = {
                    "readBytes": value["readBytes"] + to_merge["readBytes"],
                    "writeBytes": value["writeBytes"] + to_merge["writeBytes"],
                    "trafficCycles": value["trafficCycles"] + to_merge["trafficCycles"],
                }
                self.memory[key] = merged_mem
            except KeyError as exc:
                raise KeyError(
                    "Missing key in memory statistics. Cannot merge."
                ) from exc

        self.utilization.extend(op_perf_stats.utilization)
        self.sanitize_utilization_fields()


class NGPPerformanceStats:
    """Class that contains the performance stats pertaining to a model."""

    def __init__(
        self,
        debug_db: DebugDatabaseContentsType,
        performance_db: PerformanceDatabaseContentsType,
        operator_types_mapping: Union[Dict[str, str], None],
    ) -> None:
        """Initialize the class with the debug and performance database dictionaries."""
        self.debug_db: DebugDatabaseContentsType = debug_db
        self.performance_db: PerformanceDatabaseContentsType = performance_db
        self.operator_types_mapping: Union[
            Dict[str, str], None
        ] = operator_types_mapping
        self.performance_stats_per_chain: Dict[str, NGPOperatorPerformanceStats] = {}

    def process_stats_per_chain(
        self,
    ) -> dict:
        """Get the performance stats per stripe.

        Tracks a stripe to its TFLite location string
        and collates its performance statistics.

        Returns a dictionary where the key is the chain op_id
        and the value is itself a dictionary containing the
        stripe op_id, TFLite operations and statistics

        """
        for _, row in enumerate(self.performance_db):
            # find the chain ID and TF location strings

            chain_op_id, location_strings = self.track_op(stripe_op_id=str(row["id"]))

            operators = []

            for loc_str in location_strings:
                loc_str_key = loc_str[0]
                if self.operator_types_mapping:
                    op_type = self.operator_types_mapping.get(loc_str_key, "<unknown>")
                else:
                    op_type = "<unknown>"
                op_location_type = {"opLocation": loc_str, "opType": op_type}
                operators.append(op_location_type)

            # create the Operator Performance Stats object and sanitize its fields
            operator_stats = NGPOperatorPerformanceStats(
                op_id=[str(row["id"])],
                op_cycles=row["opCycles"],
                total_cycles=row["totalCycles"],
                memory=row["Memory"],
                utilization=row["Utilization"],
                operators=operators,
            )

            operator_stats.sanitize_memory_fields()
            operator_stats.sanitize_utilization_fields()

            # merge stats if that same chain has been processed already
            if chain_op_id in self.performance_stats_per_chain:
                self.performance_stats_per_chain[chain_op_id].merge(operator_stats)

            else:
                self.performance_stats_per_chain[chain_op_id] = operator_stats

        return self.performance_stats_per_chain

    def track_op(self, stripe_op_id: str) -> tuple[str, list]:
        """Track the ID of a stripe to the location string."""
        chain_op_id = self.debug_db["stripe_op_id_to_chain_op_id"][stripe_op_id]

        if len(chain_op_id) > 1:
            raise ValueError("There should be only one chain per stripe, found more!")

        fused_op_ids = self.debug_db["chain_op_id_to_fused_op_ids"][chain_op_id[0]]

        tosa_op_ids = []
        for fused_op_id in fused_op_ids:
            tosa_op_ids.extend(self.debug_db["fused_op_id_to_tosa_op_ids"][fused_op_id])

        location_strings = []
        for tosa_op_id in tosa_op_ids:
            location_strings.append(
                self.debug_db["tosa_op_id_to_api_labels"][tosa_op_id]
            )

        return chain_op_id[0], location_strings
