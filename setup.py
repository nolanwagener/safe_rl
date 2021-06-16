from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "Designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

with open(join("safe_rl", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='safe_rl',
    py_modules=['safe_rl'],
    version=__version__,#'0.1',
    install_requires=[
        'cloudpickle==1.2.1',
        'gym[atari,box2d,classic_control]',
        'ipython',
        'joblib',
        'matplotlib',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'pyyaml',
        'scipy',
        'torch==1.3.1',
        'tqdm'
    ],
)
