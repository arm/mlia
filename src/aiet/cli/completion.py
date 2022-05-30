# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Add auto completion to different shells with these helpers.

See: https://click.palletsprojects.com/en/8.0.x/shell-completion/
"""
import click


def _get_package_name() -> str:
    return __name__.split(".", maxsplit=1)[0]


# aiet completion bash
@click.group(name="completion")
def completion_cmd() -> None:
    """Enable auto completion for your shell."""


@completion_cmd.command(name="bash")
def bash_cmd() -> None:
    """
    Enable auto completion for bash.

    Use this command to activate completion in the current bash:

        eval "`aiet completion bash`"

    Use this command to add auto completion to bash globally, if you have aiet
    installed globally (requires starting a new shell afterwards):

        aiet completion bash >> ~/.bashrc
    """
    package_name = _get_package_name()
    print(f'eval "$(_{package_name.upper()}_COMPLETE=bash_source {package_name})"')


@completion_cmd.command(name="zsh")
def zsh_cmd() -> None:
    """
    Enable auto completion for zsh.

    Use this command to activate completion in the current zsh:

        eval "`aiet completion zsh`"

    Use this command to add auto completion to zsh globally, if you have aiet
    installed globally (requires starting a new shell afterwards):

        aiet completion zsh >> ~/.zshrc
    """
    package_name = _get_package_name()
    print(f'eval "$(_{package_name.upper()}_COMPLETE=zsh_source {package_name})"')


@completion_cmd.command(name="fish")
def fish_cmd() -> None:
    """
    Enable auto completion for fish.

    Use this command to activate completion in the current fish:

        eval "`aiet completion fish`"

    Use this command to add auto completion to fish globally, if you have aiet
    installed globally (requires starting a new shell afterwards):

        aiet completion fish >> ~/.config/fish/completions/aiet.fish
    """
    package_name = _get_package_name()
    print(f'eval "(env _{package_name.upper()}_COMPLETE=fish_source {package_name})"')
