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
import logging
import os
from pprint import pformat

import click
import ruamel.yaml as yaml
from configobj import ConfigObj

from crowddynamics import __version__
from crowddynamics.config import CROWDDYNAMICS_CFG
from crowddynamics.exceptions import CrowdDynamicsException, \
    NotACrowdDynamicsDirectory, DirectoryIsAlreadyCrowdDynamicsDirectory
from crowddynamics.logging import setup_logging, LOGLEVELS
from crowddynamics.parse import parse_signature, ArgSpec
from crowddynamics.simulation.multiagent import run_parallel, run_sequentially

ENVIRONMENT_YML = 'environment.yml'


class Colors:
    """Commandline output colors"""
    NEUTRAL = 'blue'
    POSITIVE = 'green'
    NEGATIVE = 'red'


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
def new(name):
    """Create new simulation. Must be called inside a crowddynamics directory.
    
    ::
    
        project
        ├── crowddynamics.cfg
        ├── environment.yml
        ├── __init__.py
        ├── <name>.py
        ...
    
    - <name>.py
        Template for making new crowd simulations.
    """
    base, ext = os.path.splitext(name)
    if ext == '.py':
        name = base
    filename = name + '.py'

    click.secho('Creating simulation: {}'.format(name), fg=Colors.NEUTRAL)

    # Check if we are inside crowddynamics project directory
    if not os.path.exists(CROWDDYNAMICS_CFG):
        click.secho('Current directory is not a simulation directory '
                    'Run "startproject" to initialise simulation.',
                    fg=Colors.NEGATIVE)
        raise NotACrowdDynamicsDirectory

    if not os.path.exists('__init__.py'):
        with open('__init__.py', 'w') as fp:
            fp.write('')

    if not os.path.exists(filename):
        # TODO: template / cookiecutter
        with open(os.path.join(filename), 'w') as fp:
            fp.write('')
        click.secho('Created successfully', fg=Colors.POSITIVE)
    else:
        click.secho('Creation failed.', fg=Colors.NEGATIVE)
        raise FileExistsError

    # Append simulation metadata to simulations sections in crowddynamics.cfg
    config = ConfigObj(CROWDDYNAMICS_CFG)
    cfg = {name: {'filename': filename}}
    try:
        config['simulations'].update(cfg)
    except KeyError:
        config['simulations'] = cfg
    config.write()


@main.command('list')
def list_of_simulations():
    """List of available simulations"""
    config = ConfigObj(CROWDDYNAMICS_CFG)
    d = config.get('simulations', [])
    click.secho('List of available simulations:', fg=Colors.NEUTRAL)
    click.secho(pformat(d), fg=Colors.NEUTRAL)


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
    setup_logging(loglevel)

    # Pass the options into the context
    context.obj = dict(
        num=num, maxiter=maxiter, timeout=timeout, profile=profile,
        parallel=parallel
    )


def mkoption(spec: ArgSpec):
    if isinstance(spec.default, int):
        return click.Option(('--' + spec.name,),
                            default=spec.default,
                            type=click.IntRange(spec.annotation[0],
                                                spec.annotation[1]))
    elif isinstance(spec.default, float):
        return click.Option(('--' + spec.name,),
                            default=spec.default,
                            type=float)
    elif isinstance(spec.default, bool):
        return click.Option(('--' + spec.name,),
                            default=spec.default,
                            is_flag=True)
    elif isinstance(spec.default, str):
        return click.Option(('--' + spec.name,),
                            default=spec.default,
                            type=click.Choice(spec.annotation))
    else:
        raise CrowdDynamicsException(
            "Option not defined for spec: {}".format(spec))


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
            processes = run_parallel(*simulations, maxiter)
            for process in processes:
                pass
        else:
            run_sequentially(*simulations, maxiter)

    # Make new command for running the simulation.
    command = click.Command(name, callback=run_simulation,
                            help=simulation.__doc__)

    # Add the command into run group
    run.add_command(command)

    # Add options for setting up the simulation
    for spec in parse_signature(simulation.set):
        command.params.append(mkoption(spec))

    return command


if __name__ == "__main__":
    main()
