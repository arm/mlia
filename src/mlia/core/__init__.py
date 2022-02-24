# Copyright (C) 2021-2022, Arm Ltd.
"""Core module.

Core module contains main components that are used in inference advisor workflow:
  - data collectors
  - data analyzers
  - advice producers
  - event publishers
  - event handlers

Inference advisor workflow consists of 3 stages:
  - data collection
  - data analysis
  - advice generation

Data is being passed from one stage to another via workflow executor.
Results (collected data, analyzed data, advice, etc) are being published via
publish/subscribe mechanishm.
"""
