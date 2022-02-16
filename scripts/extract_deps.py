# Copyright 2022, Arm Ltd.
"""Extract dependencies from setup.cfg."""
from configparser import ConfigParser
from pathlib import Path


if __name__ == "__main__":
    if not (setup_cfg := Path("setup.cfg")).is_file():
        raise Exception("setup.cfg does not exist")

    settings = [
        ("options", "install_requires", "docker/requirements.txt"),
        ("options.extras_require", "dev", "docker/requirements-dev.txt"),
    ]

    parser = ConfigParser()
    parser.read(setup_cfg)

    for section, option, output in settings:
        deps = parser.get(section, option)

        with open(output, "w") as file:
            print(f"Updating {file.name}")

            file.write("# Copyright 2021, Arm Ltd.\n")
            file.write(deps)
            file.write("\n")
