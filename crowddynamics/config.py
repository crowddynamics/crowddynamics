"""Load INI configurations using configobj module."""
from functools import lru_cache

from configobj import ConfigObj
from validate import Validator

from crowddynamics.exceptions import InvalidConfigurationError


@lru_cache()
def load_config(infile, configspec):
    """Load configuration from INI file."""
    config = ConfigObj(infile=infile, configspec=configspec)
    if not config.validate(Validator()):
        raise InvalidConfigurationError
    return config
