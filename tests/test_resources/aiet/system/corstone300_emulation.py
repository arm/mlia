# Copyright (C) 2021-2022, Arm Ltd.
"""Script for the emulation of Corstone-300 inference output."""
import random
from functools import partial


def emulate_corstone300() -> None:
    """Emulate Corstone-300 inference output."""
    random_value = partial(random.randint, 0, 1000)

    output = f"""
Corstone-300 output emulation.

NPU AXI0_RD_DATA_BEAT_RECEIVED beats: {random_value()}
NPU AXI0_WR_DATA_BEAT_WRITTEN beats: {random_value()}
NPU AXI1_RD_DATA_BEAT_RECEIVED beats: {random_value()}
NPU ACTIVE cycles: {random_value()}
NPU IDLE cycles: {random_value()}
NPU TOTAL cycles: {random_value()}
"""
    print(output)


if __name__ == "__main__":
    emulate_corstone300()
