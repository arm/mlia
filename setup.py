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
    entry_points={"console_scripts": ["mlia = mlia.main:main"]},
    scripts=[
        #        "scripts/foo.sh",
    ],
    setup_requires=["setuptools_scm"],
    install_requires=["ethos-u-vela==3.0.0"],
    maintainer="ML Inference Advisor",
    maintainer_email="matteo.martincigh@arm.com",
)
