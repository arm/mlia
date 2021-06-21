# Copyright 2021, Arm Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Module to setup the python package."""
from setuptools import find_packages
from setuptools import setup


def _readme() -> str:
    with open("README.md") as f:
        return f.read()


setup(
    name="mlia",
    use_scm_version=True,
    long_description=_readme(),
    packages=find_packages("src"),
    package_dir={"": "src"},
    entry_points={"console_scripts": ["mlia = mlia.cli.main:main"]},
    scripts=[
        #        "scripts/foo.sh",
    ],
    setup_requires=["setuptools_scm"],
    install_requires=[
        "ethos-u-vela==3.0.0",
        "typing_extensions==3.7.4",
        "numpy==1.19.5",
        "tabulate==0.8.9",
        "tensorflow==2.5.0",
        "tensorflow-model-optimization==0.5.0",
    ],
    maintainer="ML Inference Advisor",
    maintainer_email="matteo.martincigh@arm.com",
)
