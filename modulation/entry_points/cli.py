#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/02/2020
           """

import fire
import warg
from pyfiglet import Figlet

from modulation import __project__, get_version

sponsors = "Alexandra Institute"
margin_percentage = 0 / 6
terminal_width = warg.get_terminal_size().columns
margin = int(margin_percentage * terminal_width)
width = terminal_width - 2 * margin
underline = "_" * width
indent = " " * margin


class ModulationCLI(object):
    """ """

    def run(self) -> None:
        """description"""
        pass

    @staticmethod
    def version() -> None:
        """
        Prints the version of this installation."""
        draw_cli_header()
        print(f"Version: {get_version()}")

    @staticmethod
    def sponsors() -> None:
        """description"""
        print(sponsors)


def draw_cli_header(*, title=__project__, font="big"):
    """

    :param title:
    :type title:
    :param font:
    :type font:
    """
    figlet = Figlet(font=font, justify="center", width=terminal_width)
    description = figlet.renderText(title)

    print(f"{description}{underline}\n")


def main(*, always_draw_header=False):
    """

    :param always_draw_header:
    :type always_draw_header:
    """
    if always_draw_header:
        draw_cli_header()
    fire.Fire(ModulationCLI, name="modulation")


if __name__ == "__main__":
    main()
