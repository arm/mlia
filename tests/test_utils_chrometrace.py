# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for console utility functions."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from mlia.utils.chrometrace import ChromeTrace
from mlia.utils.chrometrace import parse_chrometrace
from mlia.utils.chrometrace import TraceProcess
from mlia.utils.chrometrace import TraceSlice
from mlia.utils.chrometrace import TraceThread


@pytest.fixture(name="parser")
def fixture_parser() -> ChromeTrace:
    """Create a new parser object."""
    return ChromeTrace()


@pytest.fixture(name="trace_thread")
def fixture_trace_thread() -> TraceThread:
    """Create a test TraceThread."""
    return TraceThread(22, "T22", TraceProcess(11, "P11"))


def test_ds_cnn_sum_times(test_resources_path: Path) -> None:
    """Test parsing a complete file."""
    trace_file = str(
        test_resources_path
        / "chrometrace/ds_cnn_large_fully_quantized_int8_chrome_trace.json"
    )
    trace = parse_chrometrace(Path(trace_file))

    ops_to_passes = trace.summarize_durations_per_row(
        # Pylint was complaining 're' has no 'match' member - obviously a bogus error
        # pylint: disable=no-member
        lambda slice: re.match("model", slice.name)
    )

    assert ops_to_passes == {
        "model/average_pooling2d/AvgPool->model/average_pooling2d/AvgPool_rescale": {
            (0, 0): 2.3149999999999977
        },
        "model/dense/BiasAdd->Identity_reduce_max_element_1": {(0, 0): 2.015625},
        "model/re_lu/Relu": {(0, 0): 19.771875},
        "model/re_lu_1/Relu": {(0, 1): 3.0924999999999994},
        "model/re_lu_10/Relu": {(0, 0): 9.619374999999991},
        "model/re_lu_2/Relu": {(0, 0): 9.616250000000004},
        "model/re_lu_3/Relu": {(0, 0): 3.4087499999999977},
        "model/re_lu_4/Relu": {(0, 0): 9.613750000000003},
        "model/re_lu_5/Relu": {(0, 0): 3.409374999999997},
        "model/re_lu_6/Relu": {(0, 0): 9.619374999999998},
        "model/re_lu_7/Relu": {(0, 0): 3.4093750000000043},
        "model/re_lu_8/Relu": {(0, 0): 9.623749999999994},
        "model/re_lu_9/Relu": {(0, 0): 3.410000000000011},
    }


# pylint: disable=line-too-long
def test_ds_cnn_trace(test_resources_path: Path) -> None:
    """Test parsing a complete file."""
    trace_file = str(
        test_resources_path
        / "chrometrace/ds_cnn_large_fully_quantized_int8_chrome_trace.json"
    )
    trace = parse_chrometrace(Path(trace_file))

    summary = list(
        (
            os.path.basename(str(slice.name)),
            f"#{process.name}/{thread.name}#{thread.id}",
            slice.duration,
        )
        for slice, thread, process in trace.list_records()
    )

    assert summary == [
        ("ne_convolution", "#Bound By/ne_convolution#0", 0.5106249999999999),
        ("ne_convolution", "#Bound By/ne_convolution#0", 1.0206250000000001),
        ("ne_convolution", "#Bound By/ne_convolution#0", 1.5312499999999996),
        ("ne_convolution", "#Bound By/ne_convolution#0", 3.062500000000001),
        ("ne_convolution", "#Bound By/ne_convolution#0", 1.5312499999999991),
        ("ne_convolution", "#Bound By/ne_convolution#0", 4.59375),
        ("ne_convolution", "#Bound By/ne_convolution#0", 1.53125),
        ("ne_convolution", "#Bound By/ne_convolution#0", 4.59375),
        ("ne_convolution", "#Bound By/ne_convolution#0", 1.53125),
        ("ne_vector", "#Bound By/ne_vector#1", 2.2424999999999997),
        ("ne_vector", "#Bound By/ne_vector#1", 2.2424999999999997),
        ("ne_vector", "#Bound By/ne_vector#1", 2.2424999999999997),
        ("ne_vector", "#Bound By/ne_vector#1", 2.242500000000007),
        ("pseudo_cascade_startup", "#Bound By/pseudo_cascade_startup#7", 1.021875),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            1.0537500000000009,
        ),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            1.041249999999998,
        ),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            1.0512500000000031,
        ),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            1.0418749999999974,
        ),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            1.056874999999998,
        ),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            1.0418750000000045,
        ),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            1.0612500000000011,
        ),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            1.042500000000004,
        ),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            1.056874999999991,
        ),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            0.6587499999999977,
        ),
        ("pseudo_cascade_startup", "#Bound By/pseudo_cascade_startup#7", 0.953125),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            1.0406250000000057,
        ),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            0.6512500000000045,
        ),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            0.8762499999999989,
        ),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            1.0406250000000057,
        ),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            0.8799999999999955,
        ),
        (
            "pseudo_cascade_startup",
            "#Bound By/pseudo_cascade_startup#7",
            0.6574999999999989,
        ),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("pseudo_stripe_startup", "#Bound By/pseudo_stripe_startup#8", 0.125),
        ("l2_cache", "#Bound By/l2_cache#10", 0.2718750000000014),
        ("l2_cache", "#Bound By/l2_cache#10", 0.27187499999999964),
        ("l2_cache", "#Bound By/l2_cache#10", 0.27187499999999964),
        ("l2_cache", "#Bound By/l2_cache#10", 0.27187499999999964),
        ("l2_cache", "#Bound By/l2_cache#10", 0.40749999999999886),
        ("l2_cache", "#Bound By/l2_cache#10", 0.4075000000000024),
        ("l2_cache", "#Bound By/l2_cache#10", 0.40749999999999886),
        ("l2_cache", "#Bound By/l2_cache#10", 0.40749999999999886),
        ("l2_cache", "#Bound By/l2_cache#10", 2.2424999999999997),
        ("l2_cache", "#Bound By/l2_cache#10", 2.2424999999999997),
        ("l2_cache", "#Bound By/l2_cache#10", 2.2424999999999997),
        ("l2_cache", "#Bound By/l2_cache#10", 2.242500000000007),
        ("mesh_link", "#Bound By/mesh_link#13", 0.2718750000000014),
        ("mesh_link", "#Bound By/mesh_link#13", 0.27187499999999964),
        ("mesh_link", "#Bound By/mesh_link#13", 0.27187499999999964),
        ("mesh_link", "#Bound By/mesh_link#13", 0.27187499999999964),
        ("mesh_link", "#Bound By/mesh_link#13", 0.40749999999999886),
        ("mesh_link", "#Bound By/mesh_link#13", 0.4075000000000024),
        ("mesh_link", "#Bound By/mesh_link#13", 0.40749999999999886),
        ("mesh_link", "#Bound By/mesh_link#13", 0.40749999999999886),
        ("mesh_link", "#Bound By/mesh_link#13", 2.2424999999999997),
        ("mesh_link", "#Bound By/mesh_link#13", 2.2424999999999997),
        ("mesh_link", "#Bound By/mesh_link#13", 2.2424999999999997),
        ("mesh_link", "#Bound By/mesh_link#13", 2.242500000000007),
        ("Relu", "#NE/Pass#0", 7.2718750000000005),
        ("Relu", "#NE/Pass#0", 6.249999999999999),
        ("Relu", "#NE/Pass#0", 6.25),
        ("Relu", "#NE/Pass#0", 9.616250000000004),
        ("Relu", "#NE/Pass#0", 3.4087499999999977),
        ("Relu", "#NE/Pass#0", 9.613750000000003),
        ("Relu", "#NE/Pass#0", 3.409374999999997),
        ("Relu", "#NE/Pass#0", 9.619374999999998),
        ("Relu", "#NE/Pass#0", 3.4093750000000043),
        ("Relu", "#NE/Pass#0", 9.623749999999994),
        ("Relu", "#NE/Pass#0", 3.410000000000011),
        ("Relu", "#NE/Pass#0", 9.619374999999991),
        ("AvgPool_rescale", "#NE/Pass#0", 2.3149999999999977),
        ("BiasAdd->Identity_reduce_max_element_1", "#NE/Pass#0", 2.015625),
        (
            "Identity_sub_rescale_up_1->Identity_rescale_before_table_up->Identityop6_rshift",
            "#NE/Pass#0",
            1.3531250000000057,
        ),
        (
            "Identity_op9_reducesum_element_1->Identity_op10_clz",
            "#NE/Pass#0",
            0.832499999999996,
        ),
        ("Identity_op20_sub", "#NE/Pass#0", 1.1887499999999989),
        (
            "Identity_op14_rounding_arithmetic_right_shift->Identity_op13_sub",
            "#NE/Pass#0",
            1.3531250000000057,
        ),
        ("Identity_op18_rshift->Identity_op19_mul", "#NE/Pass#0", 1.1924999999999955),
        ("Identity_op21_rshift", "#NE/Pass#0", 0.8012500000000102),
        ("Relu", "#NE/Pass#1", 1.2125000000000004),
        ("Relu", "#NE/Pass#1", 0.9400000000000013),
        ("Relu", "#NE/Pass#1", 0.9399999999999977),
        ("Identity_op11_sub->Identity_op12_lshift", "#NE/Pass#1", 0.3125),
    ]


def test_parse_sr_residual_trace(test_resources_path: Path) -> None:
    """Test parsing a complete file."""
    trace_file = str(
        test_resources_path
        / "chrometrace/sr_residual_model-e2e-conv1d_chrome_trace.json"
    )
    trace = parse_chrometrace(Path(trace_file))

    mrecs = trace.metadata_records
    assert len(mrecs) == 13
    assert len(trace.processes) == 8

    summary = list(
        (
            os.path.basename(str(slice.name)),
            f"#{process.name}/{thread.name}#{thread.id}",
            slice.duration,
        )
        for slice, thread, process in trace.list_records()
    )

    assert summary == [
        ("PreProcess", "#ShaderEngine/Pass#0", 9.106250000000001),
        ("PreProcess", "#ShaderEngine/Pass#0", 9.548125),
        ("PreProcess", "#ShaderEngine/Pass#0", 9.518749999999997),
        ("PreProcess", "#ShaderEngine/Pass#0", 7.805000000000007),
        ("PreProcess", "#ShaderEngine/Pass#0", 7.805000000000007),
        ("PreProcess", "#ShaderEngine/Pass#0", 9.546875000000028),
        ("PreProcess", "#ShaderEngine/Pass#0", 9.518749999999955),
        ("PreProcess", "#ShaderEngine/Pass#0", 7.805000000000007),
        (
            "StatefulPartitionedCall:0@VK_FORMAT_B10G11R11_UFLOAT_PACK32@",
            "#ShaderEngine/Pass#2",
            52.55375000000001,
        ),
        (
            "StatefulPartitionedCall:0@VK_FORMAT_B10G11R11_UFLOAT_PACK32@",
            "#ShaderEngine/Pass#2",
            52.55437499999999,
        ),
        (
            "StatefulPartitionedCall:0@VK_FORMAT_B10G11R11_UFLOAT_PACK32@",
            "#ShaderEngine/Pass#2",
            52.55562499999999,
        ),
        (
            "StatefulPartitionedCall:0@VK_FORMAT_B10G11R11_UFLOAT_PACK32@",
            "#ShaderEngine/Pass#2",
            52.55437499999999,
        ),
        (
            "StatefulPartitionedCall:0@VK_FORMAT_B10G11R11_UFLOAT_PACK32@",
            "#ShaderEngine/Pass#2",
            52.55687500000005,
        ),
        (
            "StatefulPartitionedCall:0@VK_FORMAT_B10G11R11_UFLOAT_PACK32@",
            "#ShaderEngine/Pass#2",
            52.55437499999999,
        ),
        (
            "StatefulPartitionedCall:0@VK_FORMAT_B10G11R11_UFLOAT_PACK32@",
            "#ShaderEngine/Pass#2",
            52.555624999999964,
        ),
        (
            "StatefulPartitionedCall:0@VK_FORMAT_B10G11R11_UFLOAT_PACK32@",
            "#ShaderEngine/Pass#2",
            52.55437499999999,
        ),
        ("FakeQuantWithMinMaxVars", "#NE/Pass#1", 14.819374999999999),
        ("FakeQuantWithMinMaxVars", "#NE/Pass#1", 12.8875),
        ("FakeQuantWithMinMaxVars", "#NE/Pass#1", 12.125),
        ("FakeQuantWithMinMaxVars", "#NE/Pass#1", 12.126249999999999),
        ("FakeQuantWithMinMaxVars", "#NE/Pass#1", 14.81812499999998),
        ("FakeQuantWithMinMaxVars", "#NE/Pass#1", 12.888125000000002),
        ("FakeQuantWithMinMaxVars", "#NE/Pass#1", 12.126875000000041),
        ("FakeQuantWithMinMaxVars", "#NE/Pass#1", 12.12624999999997),
    ]


def test_parse_mrecord_process_name(parser: ChromeTrace) -> None:
    """Test process name is picked up from metadata."""
    rec = {
        "args": {"name": "Functional Unit Utilisation (in %)"},
        "name": "process_name",
        "ph": "M",
        "pid": 1001,
    }
    parser.parse_record(rec)
    assert parser.metadata_records == [rec]
    assert parser.processes == [
        TraceProcess(1001, "Functional Unit Utilisation (in %)")
    ]


def test_parse_mrecord_thread_name(parser: ChromeTrace) -> None:
    """Test thread name is picked up from metadata."""
    recs = [
        {
            "args": {"name": "P1"},
            "name": "process_name",
            "ph": "M",
            "pid": 121,
        },
        {
            "args": {"name": "Pass"},
            "name": "thread_name",
            "ph": "M",
            "pid": 121,
            "tid": 3,
        },
    ]

    for rec in recs:
        parser.parse_record(rec)

    assert parser.metadata_records == recs
    threads = parser.get_all_threads()

    assert len(list(threads)) == 1


def test_add_thread_no_process_error(parser: ChromeTrace) -> None:
    """Check error when thread is added for missing process."""
    with pytest.raises(Exception, match="Unknown process pid=0"):
        parser.add_thread(0, 0, "T1")


def test_add_process(parser: ChromeTrace) -> None:
    """Check addition of processes."""
    parser.add_process(0, "P1")

    process = parser.get_process(0)

    assert process == TraceProcess(id=0, name="P1", threads={})

    processes = parser.processes
    assert processes == [process]


def test_add_thread(parser: ChromeTrace) -> None:
    """Check thread addition."""
    parser.add_process(3, "P1")
    process = parser.get_process(3)
    parser.add_thread(3, 6, "T1")
    procs = parser.processes
    assert len(procs) == 1

    thread = parser.get_thread(3, 6)
    assert thread == TraceThread(id=6, name="T1", parent=process)

    threads = parser.get_all_threads()
    assert threads == [thread]


def test_parse_record_invalid_type(parser: ChromeTrace) -> None:
    """Check error is reported on invalid record type."""
    rec = {
        "args": {"name": "Invalid type"},
        "name": "process_name",
        "ph": "?",
        "pid": 1001,
    }
    with pytest.raises(Exception, match="Invalid record type in chrome trace: '?'"):
        parser.parse_record(rec)


def test_thread_stack(trace_thread: TraceThread) -> None:
    """Check the behaviour of stack and duration events nesting."""
    thread = trace_thread
    assert thread.stack_depth() == 0
    thread.enter_duration_event({"ts": 1, "name": "slc0", "args": {"x": 2, "y": 8}})

    assert thread.stack_depth() == 1
    thread.enter_duration_event({"ts": 5, "name": "slice1", "args": {"y": 3}})

    assert thread.stack_depth() == 2
    thread.enter_duration_event({"ts": 10})

    assert thread.stack_depth() == 3

    ret = thread.leave_duration_event(
        {"ts": 15, "cat": "car", "args": {"x": 6, "w": 5}}
    )
    assert ret == TraceSlice(
        name=None, start=10, end=15, duration=5, args={"x": 6, "w": 5}
    )
    assert thread.stack_depth() == 2

    ret = thread.leave_duration_event({"ts": 20})
    assert ret == TraceSlice(name="slice1", start=5, end=20, duration=15, args={"y": 3})
    assert thread.stack_depth() == 1

    ret = thread.leave_duration_event({"ts": 30, "args": {"w": 5}})
    assert ret == TraceSlice(
        name="slc0", start=1, end=30, duration=29, args={"x": 2, "y": 8, "w": 5}
    )
    assert thread.stack_depth() == 0


def test_thread_stack_decreasing_ts(trace_thread: TraceThread) -> None:
    """Check invalid timestamp.

    Check that error is reported when timestamp in the end of the duration
    even is lower than at the beginning.
    """
    trace_thread.enter_duration_event({"ts": 2})
    with pytest.raises(
        Exception, match="Invalid timestamp: not increasing {'ts': 2} -> {'ts': 1}"
    ):
        trace_thread.leave_duration_event({"ts": 1})


def test_thread_stack_missing_ts(trace_thread: TraceThread) -> None:
    """Check that error reported when timestamp is missing in duration event."""
    trace_thread.enter_duration_event({})
    with pytest.raises(
        Exception, match="Timestamp missing from duration event record={}"
    ):
        trace_thread.leave_duration_event({"ts": 1})


def test_thread_stack_overflow(trace_thread: TraceThread) -> None:
    """Check that an error occurs with improper testing of duration events."""
    with pytest.raises(
        Exception,
        match="Stack empty: invalid nexting with B/E records in Chrome Trace files",
    ):
        trace_thread.leave_duration_event({})


def test_parse_duration_record(parser: ChromeTrace) -> None:
    """Check handling of duration records."""
    recs = [
        {
            "name": "myFunction",
            "cat": "foo",
            "ph": "B",
            "ts": 123,
            "pid": 2343,
            "tid": 2347,
            "args": {"first": 1},
        },
        {
            "ph": "E",
            "ts": 145,
            "pid": 2343,
            "tid": 2347,
            "args": {"first": 4, "second": 2},
        },
    ]
    parser.add_process(2343, "proc1")
    thread = parser.add_thread(2343, 2347, "thread1")
    parser.parse_events(recs)
    assert thread.slices == [
        TraceSlice(
            name="myFunction",
            start=123.0,
            end=145.0,
            duration=22.0,
            args={"first": 4, "second": 2},
        )
    ]


def test_parse_record_invalid_end(parser: ChromeTrace) -> None:
    """Test with a dangling end of a duration record."""
    rec = {
        "ph": "E",
        "ts": 145,
        "pid": 2343,
        "tid": 2347,
        "args": {"first": 4, "second": 2},
    }
    parser.add_process(2343, "proc1")
    parser.add_thread(2343, 2347, "thread1")
    with pytest.raises(
        Exception,
        match="Stack empty: invalid nexting with B/E records in Chrome Trace files",
    ):
        parser.parse_record(rec)
