# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Validate CycloneDX SBOM documents embedded in wheel files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from cyclonedx.schema import OutputFormat, SchemaVersion
from cyclonedx.validation import make_schemabased_validator


def _sbom_members(wheel: ZipFile) -> list[str]:
    """Return embedded CycloneDX JSON SBOM members."""
    members = []
    for name in wheel.namelist():
        path = PurePosixPath(name)
        if (
            len(path.parts) >= 3
            and path.parts[-2] == "sboms"
            and path.parts[-3].endswith(".dist-info")
            and path.name.endswith(".cdx.json")
        ):
            members.append(name)
    return members


def validate_wheel_sboms(wheel_path: Path) -> None:
    """Validate every CycloneDX JSON SBOM embedded in a wheel."""
    validator = make_schemabased_validator(OutputFormat.JSON, SchemaVersion.V1_6)
    with ZipFile(wheel_path) as wheel:
        members = _sbom_members(wheel)
        if not members:
            raise ValueError(f"{wheel_path} contains no CycloneDX JSON SBOM")

        for member in members:
            sbom = wheel.read(member).decode("utf-8")
            document = json.loads(sbom)
            if document.get("specVersion") != "1.6":
                raise ValueError(f"{wheel_path}:{member} is not CycloneDX 1.6")
            if error := validator.validate_str(sbom):
                raise ValueError(f"{wheel_path}:{member}: {error}")
            print(f"Validated {wheel_path}:{member}")


def main() -> None:
    """Validate SBOMs in wheels supplied on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()

    for wheel_path in args.wheels:
        validate_wheel_sboms(wheel_path)


if __name__ == "__main__":
    main()
