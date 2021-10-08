# Copyright 2021, Arm Ltd.
"""Module to setup the python package."""
from pathlib import Path
from typing import List

from setuptools import find_packages
from setuptools import setup


def _readme() -> str:
    with open("README.md") as f:
        return f.read()


def _install_requirements(requirements_filename: str = "requirements.txt") -> List[str]:
    requirements_file = Path(requirements_filename)
    if not requirements_file.exists():
        return []

    with open(requirements_file) as f:
        all_lines = (line.strip() for line in f.readlines())
        return [line for line in all_lines if line and not line.startswith("#")]


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
    install_requires=_install_requirements(),
    maintainer="ML Inference Advisor",
    maintainer_email="matteo.martincigh@arm.com",
)
