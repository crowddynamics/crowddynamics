"""Command-line interface for running crowddynamics.

References:
    - https://github.com/aharris88/awesome-cli-apps
    - https://doc.scrapy.org/en/1.3/topics/commands.html

Todo:
    - debug
    - profile
    - loglevel
    - help text for mkoption
    - Launch multiple simulation in parallel
    - Scheduler
    - Iterations limit
    - Time limit
    - Simulation time limit
    - Read simulation configs from file
    - load and restart old simulations
    - Progress bar
    - random seed
    - chaining command to run different simulation types at chained, should
      also work in parallel
    - token normalization

"""
import inspect
import os
from collections import namedtuple
from functools import partial
from pprint import pformat

import click
import logging

import crowddynamics
from crowddynamics.exceptions import CrowdDynamicsException, InvalidArgument
from crowddynamics.logging import setup_logging, user_info
from crowddynamics.multiagent import examples
from crowddynamics.multiagent.simulation import REGISTERED_SIMULATIONS, \
    run_simulations_parallel, run_simulations_sequentially
from crowddynamics.plugins.gui import run_gui

VERSION = crowddynamics.__version__
HELP = "CrowdDynamics {version}. A tool for building and running crowd " \
       "simulations.".format(version=VERSION)
LOGLEVELS = [
    logging.CRITICAL,
    logging.FATAL,
    logging.ERROR,
    logging.WARNING,
    logging.WARN,
    logging.INFO,
    logging.DEBUG,
    logging.NOTSET,
]
# TODO: loglevel by name
LOGLEVELS += ['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'WARN', 'INFO',
              'DEBUG', 'NOTSET', ]

examples.init()


@click.group(help=HELP)
@click.version_option(VERSION)
def main():
    pass


@main.command()
@click.argument('name')
def startproject(name):
    """Create new simulation project."""
    # TODO: Handle lower / upper case
    if os.path.exists(name):
        click.secho('Project "{name}" already exists.'.format(name=name),
                    fg='red')
        return
    click.secho('Creating project: {name}'.format(name=name), fg='green')
    os.mkdir(name)
    # TODO: project template


@main.command()
@click.argument('name')
def gensimulation(name):
    """Create new simulation."""
    # TODO: simulation template
    pass


@main.command()
def list():
    """List of available simulations"""
    # TODO: pretty formatting, colors
    click.secho('List of available simulations:', fg='green')
    click.secho(pformat(REGISTERED_SIMULATIONS), fg='green')


@main.group(chain=True)
@click.option('--loglevel', type=click.Choice(LOGLEVELS),
              default=logging.INFO,
              help='Choices for setting logging level.')
@click.option('--num', type=int, default=1,
              help='Number of simulations to run.')
@click.option('--maxiter', type=int, default=None,
              help='Maximum number of iterations')
@click.option('--timeout', type=float, default=None,
              help='Time limit for the simulation.')
@click.option('--parallel', is_flag=True, default=False,
              help='If parallel flag is set simulations will be run in '
                   'multiple processes.')
@click.option('--profile', is_flag=True,
              help='Flag whether to profile the simulation.')
@click.pass_context
def run(context, loglevel, num, maxiter, timeout, parallel, profile):
    """Run simulation from the command-line."""
    # TODO: set loglevel
    setup_logging()
    user_info()

    # Pass the options into the context
    context.obj = dict(
        num=num, maxiter=maxiter, timeout=timeout, profile=profile,
        parallel=parallel
    )


@main.command()
def gui():
    """Run graphical user interface."""
    # TODO: move to `plugins.gui`
    setup_logging()
    user_info()
    run_gui()


ArgSpec = namedtuple('ArgSpec', ('name', 'default', 'type', 'annotation'))


def mkspec(parameter):
    if isinstance(parameter.default, inspect.Parameter.empty):
        raise InvalidArgument('Default argument should not be empty.')
    return ArgSpec(name=parameter.name,
                   default=parameter.default,
                   type=type(parameter.default),
                   annotation=parameter.annotation)


def parse_signature(function):
    """Parse signature

    .. list-table::
       :header-rows: 1

       * - Type
         - Validation
         - Click option
         - Qt widget
       * - int
         - interval
         - IntRange
         - QSpinBox
       * - float
         - interval
         - float with callback
         - QDoubleSpinBox
       * - bool
         - flag
         - Boolean flag
         - QRadioButton
       * - str
         - choice
         - Choice
         - QComboBox

    Args:
        function:

    Yields:
        ArgSpec:

    """
    sig = inspect.signature(function)
    for name, p in sig.parameters.items():
        if name != 'self':
            yield mkspec(p)


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


def mkcommand(name, simulation):
    """Make command

    Args:
        name (str):
        simulation (MultiAgentSimulation):

    Returns:
        click.Command:

    Todo:
        - make lazy
        - run sequentially / parallel
    """

    @click.pass_context
    def run_simulation(context, *args, **kwargs):
        """"Callback function that is called when the command is executed."""
        # click.echo(context.obj)
        # click.echo(args)
        # click.echo(kwargs)

        num = context.obj['num']
        maxiter = context.obj['maxiter']

        # Initialise the ``num`` simulations
        simulations = []
        for _ in range(num):
            simu = simulation()
            simu.set(*args, **kwargs)
            simulations.append(simu)

        # Run options from the context
        if num > 1 and context.obj['parallel']:
            processes = run_simulations_parallel(simulations, maxiter)
            for process in processes:
                pass
        else:
            run_simulations_sequentially(simulations, maxiter)

    # Make new command for running the simulation.
    command = click.Command(name, callback=run_simulation,
                            help=simulation.__doc__)

    # Add the command into run group
    run.add_command(command)

    # Add options for setting up the simulation
    for spec in parse_signature(simulation.set):
        command.params.append(mkoption(spec))

    return command


for name, simulation in REGISTERED_SIMULATIONS.items():
    mkcommand(name, simulation)

if __name__ == "__main__":
    main()
