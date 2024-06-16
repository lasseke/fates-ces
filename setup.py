"""Specifies pip installation details"""

from setuptools import find_packages, setup

DESCRIPTION_STR = '''
Site-level calibration of the Functionally Assembled Terrestrial
Ecosystem Simulator (FATES) embedded in the Community Land Model
(CLM).
'''

setup(
    name='fatescal',
    packages=find_packages(include=['fatescal', 'fatescal.*']),
    version='0.1.0',
    description=DESCRIPTION_STR,
    author='Lasse T. Keetz, Kristoffer Aalstad, Rosie A. Fisher',
    keywords=[
        "Dynamic Global Vegetation Model", "Earth System Model",
        "Functionally Assembled Terrestrial Ecosystem Simulator",
        "FATES", "CLM", "Land surface model", "Data assimilation",
        "Machine learning",
    ],
    cmdclass={'bdist_wheel': None},
)
