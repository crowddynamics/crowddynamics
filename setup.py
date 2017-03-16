import os
from setuptools import setup, find_packages
import versioneer


def readme():
    with open('README.rst') as f:
        return f.read()


def license():
    with open('LICENSE') as f:
        return f.read()


def requirements(name):
    """Parse requirements from file inside requirements directory. Does not
    hander ``-r file.txt`` syntax."""
    with open(name) as f:
        lines = []
        while True:
            line = f.readline()
            if line == '\n' or line.startswith('#') or line.startswith('-'):
                continue
            if line == '':
                break
            lines.append(line)
        return lines


setup(
    name='crowddynamics',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Python package for simulating, visualizing and analysing '
                'movement of human crowds.',
    long_description=readme(),
    author='Jaan Tollander de Balsch',
    author_email='de.tollander@aalto.fi',
    url='https://github.com/jaantollander/CrowdDynamics',
    license=license(),
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'crowddynamics=crowddynamics.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=[],
    zip_safe=False,
    keywords='',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='crowddynamics.tests',
    test_requirements=[],
)
