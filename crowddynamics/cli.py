"""Command-line interface for running crowddynamics.

Todo:
    - debug
    - loglevel
"""
import click
import colorama

import crowddynamics
from crowddynamics.exceptions import CrowdDynamicsException
from crowddynamics.logging import setup_logging, user_info
from crowddynamics.multiagent import examples
from crowddynamics.multiagent.simulation import REGISTERED_SIMULATIONS
from crowddynamics.plugins.gui import run_gui
from crowddynamics.validation import parse_signature

colorama.init()  # Enable colors
examples.init()


@click.group()
@click.version_option(crowddynamics.__version__)
def main():
    """Main commands."""
    setup_logging()
    user_info()


@main.group()
def run():
    """Run simulation from the command-line."""
    pass


@main.command()
def gui():
    """Run graphical user interface."""
    run_gui()


def mkoption(spec):
    if spec.type is int:
        return click.Option(('--' + spec.name,),
                            default=spec.default,
                            type=click.IntRange(spec.annotation[0],
                                                spec.annotation[1]))
    elif spec.type is float:
        return click.Option(('--' + spec.name,),
                            default=spec.default,
                            type=float)
    elif spec.type is bool:
        return click.Option(('--' + spec.name,),
                            default=spec.default,
                            is_flag=True)
    elif spec.type is str:
        return click.Option(('--' + spec.name,),
                            default=spec.default,
                            type=click.Choice(spec.annotation))
    else:
        raise CrowdDynamicsException(
            "Option not defined for spec: {}".format(spec)
        )


def mkcommand(simulation):
    # TODO: help
    # Callback function that is called when the command is executed
    def callback(*args, **kwargs):
        simu = simulation()
        simu.set(*args, **kwargs)

    # Command
    name = simulation.__name__
    cmd = click.Command(name, callback=callback)
    run.add_command(cmd)

    # Options
    for spec in parse_signature(simulation.set):
        cmd.params.append(mkoption(spec))


for simu in REGISTERED_SIMULATIONS:
    mkcommand(simu)

if __name__ == "__main__":
    main()
