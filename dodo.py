#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import shutil

import functools

try:
    from pathlib import Path
except ImportError():
    # If using python version < 3.4
    from pathlib2 import Path

# Logging
logger = logging.getLogger(__name__)

# Configurations and metadata
AUTHOR = ''
PROJECT = ''
GITHUB_REPO = ''


# Sphinx
SPHINXOPTS = ''
SOURCEDIR = 'docs'
BUILDDIR = '_build'
APIDOCSDIR = os.path.join(SOURCEDIR, 'apidocs')

# -----------------------------------------------------------------------------
# doit configurations
# -----------------------------------------------------------------------------
DOIT_CONFIG = {
    'default_tasks': [],
    'verbosity': 0,
}


def set_default_task(task):
    """Decorator for setting task into default tasks
    >>> @set_default_task
    >>> def task_do_something():
    >>>     ...
    would do same as
    >>> DOIT_CONFIG['default_tasks'].append('do_something')
    """

    @functools.wraps(task)
    def wrapper(*args, **kwargs):
        name = task.__name__.strip('task_')
        globals().setdefault('DOIT_CONFIG', dict())
        DOIT_CONFIG.setdefault('default_tasks', list())
        DOIT_CONFIG['default_tasks'].append(name)
        result = task(*args, **kwargs)
        return result

    return wrapper


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------


def create_files(*paths):
    """Create files and folders.
    Examples:
        >>> create_files('file.txt', 'file2.txt')
    Args:
        *paths (str|Path):
    Todo:
        - write content
    """
    for filepath in map(Path, paths):
        dirname, _ = os.path.split(str(filepath))
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        filepath.touch(exist_ok=True)


def remove_files(*paths):
    """Remove files and folders. Supports unix style glob syntax.
    Examples:
        >>> remove_files('file.txt', '**/file.txt', 'file.*')
    Args:
        *paths (str|Path):
    """
    p = Path('.')
    for path in paths:
        for pathname in p.glob(str(path)):
            try:
                if pathname.is_dir():
                    shutil.rmtree(str(pathname))
                else:
                    os.remove(str(pathname))
            except FileNotFoundError:
                pass


def combine(*tasks):
    """Combine actions of different tasks
    Examples:
        >>> combine({'actions': ['action1']}, {'actions': ['action2']})
        {'actions': ['action1', 'action2']}
    Args:
        *tasks (dict): Tasks that containing actions to be combined
    """
    return {'actions': sum((task.get('actions', []) for task in tasks), [])}


# -----------------------------------------------------------------------------
# doit tasks
# -----------------------------------------------------------------------------


def task_clean_build():
    """Clean build artifacts"""
    files = ['build/', 'dist/', '.eggs/', '*.egg-info', '*.egg']
    return {'actions': [(remove_files, files)]}


def task_clean_pyc():
    """Clean python file artifacts"""
    # TODO: recursive
    files = ['*.pyc', '*.pyo', '*~', '__pycache__']
    return {'actions': [(remove_files, files)]}


def task_clean_test():
    """Clean test and coverage artifacts"""
    files = ['.tox/', '.coverage', 'htmlcov/', '.benchmarks', '.cache',
             '.hypothesis']
    return {'actions': [(remove_files, files)]}


def task_clean_docs():
    """Clean documentation"""
    files = [os.path.join(SOURCEDIR, BUILDDIR)]
    return {'actions': [(remove_files, files)]}


def task_clean_apidocs():
    """Clean documentation"""
    files = [os.path.join(SOURCEDIR, 'apidocs'),
             os.path.join(SOURCEDIR, 'apidocs_examples')]
    return {'actions': [(remove_files, files)]}
