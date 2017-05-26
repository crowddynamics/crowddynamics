"""Command-line interface for running crowddynamics."""
import logging
import os
from collections import OrderedDict
from pprint import pformat

import click

from crowddynamics import __version__
from crowddynamics.logging import setup_logging, LOGLEVELS
from crowddynamics.simulation.multiagent import MultiAgentSimulation
from crowddynamics.utils import import_subclasses


class Colors:
    """Color palette for the commandline output colors"""
    NEUTRAL = 'blue'
    POSITIVE = 'green'
    NEGATIVE = 'red'


def import_simulations():
    """Import simulations from modules of the current working directory."""
    d = OrderedDict()
    for path in os.listdir('.'):
        base, ext = os.path.splitext(path)
        if ext == '.py':
            d.update(import_subclasses(path, MultiAgentSimulation))
    return d


def trait_to_option():
    pass


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

    # Pass the options into the context
    context.obj = dict(num=num)


if __name__ == "__main__":
    main()
