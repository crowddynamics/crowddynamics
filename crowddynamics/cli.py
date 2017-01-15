"""
Command-line interface for running ``crowddynamics``.
"""
import click

from crowddynamics.functions import setup_logging, user_info
from crowddynamics.gui.run import run_gui


# TODO: Add colors to commands


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
