"""CrowdDynamics simulation configuration"""
import importlib.util
import logging
import os

from configobj import ConfigObj
from validate import Validator

from crowddynamics.exceptions import ValidationError, deprecated
from crowddynamics.utils import remember_cwd

CONFIG_ROOT = os.path.join(os.path.dirname(__file__), 'conf')

LOG_CFG = os.path.join(CONFIG_ROOT, 'logging.yaml')

CROWDDYNAMICS_CFG_SPEC = os.path.join(CONFIG_ROOT, 'crowddynamics_spec.cfg')
CROWDDYNAMICS_CFG = 'crowddynamics.cfg'

BODY_TYPES_CFG = os.path.join(CONFIG_ROOT, 'body_types.cfg')
BODY_TYPES_CFG_SPEC = os.path.join(CONFIG_ROOT, 'body_types_spec.cfg')


def load_config(infile, configspec=None):
    """Load configuration from INI file."""
    config = ConfigObj(infile=infile, configspec=configspec)
    if configspec and not config.validate(Validator()):
        raise ValidationError
    return config


@deprecated
def import_simulation_callables(confpath):
    """Import simulations callables
    
    https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

    Args:
        confpath (str|Path):

    Yields:
        (str, typing.Callable[[...], MultiAgentSimulation]):
    """
    logger = logging.getLogger(__name__)
    base, _ = os.path.split(confpath)
    config = load_config(confpath, configspec=CROWDDYNAMICS_CFG_SPEC)
    conf = config.get('simulations', [])

    if conf:
        for name, configs in conf.items():
            path = configs['path']
            for funcname in configs['functions']:
                try:
                    with remember_cwd(base):
                        spec = importlib.util.spec_from_file_location(name, path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        try:
                            yield funcname, getattr(module, funcname)
                        except AttributeError as e:
                            logger.warning(e)
                except ImportError as e:
                    logger.warning(e)
