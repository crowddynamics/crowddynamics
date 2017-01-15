from setuptools import setup, find_packages
import versioneer


def readme():
    with open('README.md') as f:
        return f.read()


def license():
    with open('LICENSE') as f:
        return f.read()


setup(
    name='crowddynamics',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='',
    long_description=readme(),
    author='Jaan Tollander de Balsch',
    author_email='de.tollander@aalto.fi',
    url='https://github.com/jaantollander/CrowdDynamics',
    license=license(),
    packages=find_packages()
)
