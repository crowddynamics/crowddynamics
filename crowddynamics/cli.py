"""Command-line interface for running crowddynamics.

Todo:
    - Colors
"""
import click
import colorama

from crowddynamics.logging import setup_logging, user_info
from crowddynamics.plugins.gui import run_gui


# Enable colors on windows
colorama.init()


@click.group()
def main():
    """Main commands."""
    setup_logging()
    user_info()


@main.command()
def run():
    """Run simulation from the command-line."""
    pass


@main.command()
def gui():
    """Run graphical user interface."""
    run_gui()


if __name__ == "__main__":
    main()
