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
import logging
import os
import platform
import sys
from collections import namedtuple
from pprint import pformat

import click
from loggingtools import setup_logging
import ruamel.yaml as yaml
from configobj import ConfigObj

from crowddynamics import __version__
from crowddynamics.exceptions import CrowdDynamicsException, InvalidArgument, \
    NotACrowdDynamicsDirectory, DirectoryIsAlreadyCrowdDynamicsDirectory
from crowddynamics.multiagent import examples
from crowddynamics.multiagent.simulation import REGISTERED_SIMULATIONS, \
    run_simulations_parallel, run_simulations_sequentially


class Colors:
    """Commandline output colors"""
    NEUTRAL = 'blue'
    POSITIVE = 'green'
    NEGATIVE = 'red'


CROWDDYNAMICS_CFG = 'crowddynamics.cfg'
ENVIRONMENT_YML = 'environment.yml'

LOGLEVELS = [
    logging.CRITICAL, logging.FATAL, logging.ERROR, logging.WARNING,
    logging.WARN, logging.INFO, logging.DEBUG, logging.NOTSET,
    'CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'WARN', 'INFO', 'DEBUG', 'NOTSET'
]
BASE_DIR = os.path.dirname(__file__)
LOG_CFG = os.path.join(BASE_DIR, 'logging.yaml')

examples.init()


def user_info():
    logger = logging.getLogger(__name__)
    logger.info("Platform: %s", platform.platform())
    logger.info("Path: %s", sys.path[0])
    logger.info("Python: %s", sys.version[0:5])


@click.group(help="CrowdDynamics {version}. A tool for building and running "
                  "crowd simulations.".format(version=__version__))
@click.version_option(__version__)
def main():
    pass


@main.command()
@click.argument('dirpath')
def startproject(dirpath):
    """Create new simulation project.
    
    ::
    
        <name>
        ├── crowddynamics.cfg
        ├── environment.yml
        ├── ...
        
    - crowddynamics.cfg
        Configurations and path to simulations.
    - environment.yml
        Conda environment to install and run simulations.
    
    Args:
        dirpath (str):
          - ``.``: Current directory
          - ``path/to/name``: Path to directory called ``name``  
    """
    click.secho('Creating project: {name}'.format(name=dirpath),
                fg=Colors.NEUTRAL)

    if os.path.exists(os.path.join(dirpath, CROWDDYNAMICS_CFG)):
        click.secho('Dirpath is already crowddynamics directory.',
                    fg=Colors.NEGATIVE)
        raise DirectoryIsAlreadyCrowdDynamicsDirectory

    try:
        # Project directory
        os.makedirs(dirpath, exist_ok=True)
        click.secho('Created successfully', fg=Colors.POSITIVE)
    except FileExistsError as error:
        click.secho('Creation failed', fg=Colors.NEGATIVE)
        raise error

    # crowddynamics.cfg
    config = ConfigObj(os.path.join(dirpath, CROWDDYNAMICS_CFG))
    config['simulations'] = {}
    config.write()

    # environment.yml
    data = {'channels': ['conda-forge'],
            'dependencies': ['crowddynamics']}
    with open(os.path.join(dirpath, ENVIRONMENT_YML), 'w') as envfile:
        yaml.dump(data, envfile, default_flow_style=False, indent=2)


@main.command()
@click.argument('name')
def newsimulation(name):
    """Create new simulation. Must be called inside a crowddynamics directory.
    
    ::
    
        project
        ├── crowddynamics.cfg
        ├── environment.yml
        ├── <name>
        │   ├── __init__.py
        │   └── simulation.py
        ...
    
    - simulation.py
        Template for making new crowd simulations.
    """
    click.secho('Creating simulation: {}'.format(name), fg=Colors.NEUTRAL)

    # Check if we are inside crowddynamics project directory
    if not os.path.exists(CROWDDYNAMICS_CFG):
        click.secho('Current directory is not a simulation directory',
                    fg=Colors.NEGATIVE)
        raise NotACrowdDynamicsDirectory

    try:
        os.mkdir(name)
        click.secho('Created successfully', fg=Colors.POSITIVE)
    except FileExistsError as error:
        click.secho('Creation failed.', fg=Colors.NEGATIVE)
        raise error

    # Append simulation metadata to simulations sections in
    # crowddynamics.cfg
    config = ConfigObj(CROWDDYNAMICS_CFG)
    cfg = {name: {'path': name}}
    try:
        config['simulations'].update(cfg)
    except KeyError:
        config['simulations'] = cfg
    config.write()

    # Make Python simulation template
    with open(os.path.join(name, '__init__.py'), 'w') as fp:
        fp.write('')

    with open(os.path.join(name, 'simulation.py'), 'w') as fp:
        fp.write('')


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
    setup_logging(LOG_CFG)
    user_info()

    # Pass the options into the context
    context.obj = dict(
        num=num, maxiter=maxiter, timeout=timeout, profile=profile,
        parallel=parallel
    )


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
