# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Core module.

Core module contains the main components that are used in the workflow of
ML Inference Advisor:
  - data collectors
  - data analyzers
  - advice producers
  - event publishers
  - event handlers

The workflow of ML Inference Advisor consists of 3 stages:
  - data collection
  - data analysis
  - advice generation

Data is being passed from one stage to another via workflow executor.
Results (collected data, analyzed data, advice, etc) are being published via
publish/subscribe mechanishm.
"""
