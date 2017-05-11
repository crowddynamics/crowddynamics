from setuptools import setup, find_packages

import versioneer


def readfile(filepath):
    with open(filepath) as f:
        return f.read()


setup(
    name='crowddynamics',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Python package for simulating, visualizing and analysing '
                'movement of human crowds.',
    long_description=readfile('README.rst'),
    author='Jaan Tollander de Balsch',
    author_email='de.tollander@aalto.fi',
    url='https://github.com/jaantollander/crowddynamics',
    license=readfile('LICENSE.txt'),
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'crowddynamics=crowddynamics.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=[],
    extra_require={
        'docs': ['sphinx',
                 'sphinx-rtd-theme',
                 'graphviz',
                 'sphinxcontrib-tikz'],
        'tests': ['pytest',
                  'pytest-cov',
                  'pytest-benchmark',
                  'hypothesis'],
    },
    zip_safe=False,
    keywords='',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
