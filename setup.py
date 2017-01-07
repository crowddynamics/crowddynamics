from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()


setup(
    name='CrowdDynamics',
    version='0.1',
    description='',
    long_description=readme,
    author='Jaan Tollander de Balsch',
    author_email='de.tollander@aalto.fi',
    url='https://github.com/jaantollander/CrowdDynamics',
    license=license,
    packages=find_packages()
)
