"""
Configuration for tests.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = "output"

# TODO: pytest, pytest-plugins

# TODO: Hypothesis settings
# https://hypothesis.readthedocs.io/en/latest/settings.html

# TODO: Floating point numbers
# https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
EPSILON = sys.float_info.epsilon

