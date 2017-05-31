"""Command-line interface for running crowddynamics."""
import logging
import os
from collections import OrderedDict
from pprint import pformat

import click

from crowddynamics import __version__
from crowddynamics.logging import setup_logging, LOGLEVELS
from crowddynamics.simulation.multiagent import MultiAgentSimulation
from crowddynamics.traits import class_own_traits, \
    trait_to_option
from crowddynamics.utils import import_subclasses


class Colors:
    """Color palette for the commandline output colors"""
    NEUTRAL = 'blue'
    POSITIVE = 'green'
    NEGATIVE = 'red'


def import_simulations(dir_path='.'):
    """Import simulations from modules of the current working directory."""
    d = OrderedDict()
    for path in os.listdir(dir_path):
        base, ext = os.path.splitext(path)
        if ext == '.py':
            d.update(import_subclasses(path, MultiAgentSimulation))
    return d


@click.group(help="CrowdDynamics {version}. A tool for building and running "
                  "crowd simulations.".format(version=__version__))
@click.version_option(__version__)
def main():
    pass


@main.command('list')
def list_of_simulations():
    """List of available simulations"""
    d = import_simulations()
    click.secho('List of available simulations:', fg=Colors.NEUTRAL)
    click.secho(pformat(d), fg=Colors.POSITIVE)


@main.command()
@click.option('--directory', '-d', default='.')
@click.option('--basename', '-n')
def concat_npy(directory, basename):
    import numpy as np
    from crowddynamics.io import load_npy_concatenated
    path = os.path.abspath(directory)
    arr = load_npy_concatenated(path, basename)
    np.save(os.path.join(path, basename + '.npy'), arr)


@main.group(chain=True)
@click.option('--loglevel', type=click.Choice(LOGLEVELS),
              default=logging.INFO,
              help='Choices for setting logging level.')
@click.option('--num', type=int, default=1,
              help='Number of simulations to run.')
@click.pass_context
def run(context, loglevel, num):
    """Run simulation from the command-line."""
    setup_logging(loglevel)
    context.obj = dict(num=num)


def simulation_commands():
    """Make commands simulations imported from python files in current working 
    directory."""
    simulations = import_simulations(dir_path='.')
    for simulation_name, simulation_cls in simulations.items():
        # New command
        command = click.Command(
            simulation_name, callback=lambda: simulation_cls().run())

        # Add the command into run group
        run.add_command(command)

        # Add options for setting up the simulation
        for name, trait in class_own_traits(simulation_cls):
            command.params.append(trait_to_option(name, trait))


if __name__ == "__main__":
    simulation_commands()
    main()
