# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Utility to parse Chrome Event Trace files."""
from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterable

# Keys in TraceEvent format
#
# name: The name of the event, as displayed in Trace Viewer
# cat:  The event categories. This is a comma separated list of categories for
#       the event. The categories can be used to hide events in the Trace Viewer UI.
# ph:   The event type. This is a single character which changes depending on the type
#       of event being output. The valid values are listed in the table below.
# ts:   The tracing clock timestamp of the event. The timestamps are provided at
#       microsecond granularity.
# tts:  Optional. The thread clock timestamp of the event. The timestamps are provided
#       at microsecond granularity.
# pid:  The process ID for the process that output this event.
# tid:  The thread ID for the thread that output this event.
# args: Any arguments provided for the event. Some of the event types have required
#       argument fields, otherwise, you can put any information you wish in here.
#       The arguments are displayed in Trace Viewer when you view an event in the
#       analysis section.
#
# For more details:
# * https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU


@dataclass
class TraceProcess:
    """Represents a process in the Chrome TraveEvent data model."""

    # pylint: disable=invalid-name
    id: int
    name: str
    threads: dict[int, TraceThread] = field(default_factory=dict)


@dataclass
class TraceThread:
    """Keeps information about a thread object in the event trace format."""

    # pylint: disable=invalid-name
    id: int
    name: str
    parent: TraceProcess
    stack: list[dict] = field(default_factory=list)
    slices: list[TraceSlice] = field(default_factory=list)

    def enter_duration_event(self, record: dict) -> None:
        """Record the start record of a duration event."""
        self.stack.append(record)

    def leave_duration_event(self, record2: dict) -> TraceSlice:
        """Record the end record of a duration event."""
        if self.stack_depth() < 1:
            raise RuntimeError(
                "Stack empty: invalid nexting with B/E records in Chrome Trace files"
            )
        record1 = self.stack.pop()
        args1 = record1.get("args")
        args2 = record2.get("args")

        # Merging all new keys into the old map (new ones take precendence)
        record3 = {}
        record3.update(record1)
        record3.update(record2)

        def get_timestamp(record: dict[str, Any]) -> float:
            """Extract the timestamp from an event."""
            timestamp = record.get("ts")
            if timestamp is None:
                raise KeyError(f"Timestamp missing from duration event {record=}")
            return float(timestamp)

        ts1 = get_timestamp(record1)
        ts2 = get_timestamp(record2)
        if ts1 >= ts2:
            raise ValueError(
                f"Invalid timestamp: not increasing {record1} -> {record2}"
            )

        # Handling args differently, to other keys; doing the merging separately on that
        args = {}
        for arguments in args1, args2:
            if arguments is not None:
                args.update(arguments)

        slc = TraceSlice(
            name=record3.get("name"), start=ts1, end=ts2, duration=ts2 - ts1, args=args
        )
        self.slices.append(slc)
        return slc

    def stack_depth(self) -> int:
        """Return the depth of the nesting of duration events."""
        return len(self.stack)


@dataclass
class TraceSlice:
    """Holds information about a slice in the TraceEvent model."""

    name: str | None
    start: float
    end: float
    duration: float
    args: dict


class ChromeTrace:
    """
    Chrome Trace Event parser.

    Load, parse, and extract basic information out of a
    Chrome Trace Event JSON file.
    """

    all_events: list[dict] = []
    _metadata_records: list[dict] = []
    _processes: dict[int, TraceProcess]

    def __init__(self) -> None:
        """Initialize fields common properties."""
        self._metadata_records: list[dict] = []
        self._processes: dict[int, TraceProcess] = {}

    def parse_file(self, trace_file: Path) -> None:
        """Parse a JSON Trace Event file."""
        with open(trace_file, encoding="utf-8") as read_file:
            data = json.load(read_file)
            events = data["traceEvents"]
            self.parse_events(events)

    def add_process(self, pid: int, name: str) -> None:
        """Add process with give id and name."""
        self._processes[pid] = TraceProcess(pid, name)

    def get_process(self, pid: int) -> TraceProcess:
        """Find process with given id."""
        process = self._processes.get(pid)
        if process is None:
            raise ValueError(f"Unknown process {pid=}.")
        return process

    def add_thread(self, pid: int, tid: int, name: str) -> TraceThread:
        """Add thread with given process and thread ids."""
        process = self.get_process(pid)
        thread = TraceThread(tid, name, process)
        process.threads[tid] = thread
        return thread

    def get_thread(self, pid: int, tid: int) -> TraceThread:
        """Find thread for given process and thread ids."""
        process = self.get_process(pid)
        thread = process.threads.get(tid)
        if thread is None:
            raise ValueError(f"Unknown thread {pid=} {tid=}.")
        return thread

    def parse_record(self, rec: dict[str, Any]) -> None:
        """Parse one dict object following the Trace event data structure."""
        rectype: str = rec["ph"]
        recname: str | None = rec.get("name")
        pid: int = rec["pid"]
        tid: int = cast(int, rec.get("tid"))
        args = cast(dict, rec.get("args"))
        if rectype == "M":
            self._metadata_records.append(rec)
            if args is None:
                raise ValueError(f"Invalid process name record {rec}")
            name = args["name"]
            if recname == "process_name":
                self.add_process(pid, name)
            elif recname == "thread_name":
                if tid is None:
                    raise ValueError("Metadata for thread name is missing thread id")
                self.add_thread(pid, tid, name)
            else:
                raise ValueError(f"Unknown metadata record name: '{recname}' {rec}")
        elif rectype == "B":
            thread = self.get_thread(pid, tid)
            thread.enter_duration_event(rec)
        elif rectype == "E":
            thread = self.get_thread(pid, tid)
            thread.leave_duration_event(rec)
        elif rectype == "C":
            pass
        else:
            raise ValueError(f"Invalid record type in chrome trace: '{rectype}'")

    def parse_events(self, events: list[dict]) -> None:
        """Parse all the event passed as a list of dicts in Trace Event structure."""
        for event in events:
            self.parse_record(event)

    @property
    def metadata_records(self) -> list[dict]:
        """Return all the metadata records."""
        return self._metadata_records

    @property
    def processes(self) -> list[TraceProcess]:
        """Return all processes as a property."""
        return list(self._processes.values())

    def get_all_threads(self) -> Iterable[TraceThread]:
        """Return all the threads."""
        threads = [p.threads.values() for p in self._processes.values()]

        def flatten(lst: list[Any]) -> Iterable[TraceThread]:
            return [item for sublist in lst for item in sublist]

        return flatten(threads)

    def list_records(self) -> Iterable[tuple[TraceSlice, TraceThread, TraceProcess]]:
        """
        Iterate through the event trace model.

        Returns a generator that iterates through all collected trace records,
        that is, slices with the corresponding threads and processes.
        """
        for thread in self.get_all_threads():
            for slc in thread.slices:
                yield (slc, thread, thread.parent)

    def summarize_durations_per_row(
        self, slice_filter: Callable = lambda: True
    ) -> dict[str, dict[tuple[int, int], float]]:
        """
        Build a map summarizing the time spent in each slice.

        Returns mapping from slices to the sum of their durations
        per process/thread.
        """
        slice_row_times: dict[str, dict[tuple[int, int], float]] = {}
        for slc, thread, process in self.list_records():
            if not slice_filter(slc):
                continue

            slice_name = cast(str, slc.name)
            row_times = slice_row_times.get(slice_name, {})
            row = (process.id, thread.id)
            time = row_times.get(row, 0.0)
            time += slc.duration
            row_times[row] = time
            slice_row_times[slice_name] = row_times
        return slice_row_times

    def get_pass_times(self, slice_filter: Callable = lambda: True) -> dict[int, float]:
        """Get total times spent in each pass."""
        pass_times: dict[int, float] = {}
        for slc, thread, _ in self.list_records():
            if not slice_filter(slc):
                continue

            pass_time = pass_times.get(thread.id, 0)
            pass_time += slc.duration
            pass_times[thread.id] = pass_time

        return pass_times


def parse_chrometrace(trace_file: Path) -> ChromeTrace:
    """Parse the trace file and return the parser object."""
    trace = ChromeTrace()
    trace.parse_file(trace_file)
    return trace
