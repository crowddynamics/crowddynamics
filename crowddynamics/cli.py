"""Command-line interface for running crowddynamics."""
import click
import colorama

from crowddynamics.logging import setup_logging, user_info
from crowddynamics.plugins.gui import run_gui

from crowddynamics.multiagent import examples
from crowddynamics.multiagent.simulation import REGISTERED_SIMULATIONS


# Enable colors on windows
colorama.init()


@click.group()
def main():
    """Main commands."""
    setup_logging()
    user_info()
    examples.init()


@main.command()
def run():
    """Run simulation from the command-line."""
    pass


for simulation in REGISTERED_SIMULATIONS:
    pass



@main.command()
def gui():
    """Run graphical user interface."""
    run_gui()


if __name__ == "__main__":
    main()
