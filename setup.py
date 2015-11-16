#!/usr/bin/env python
#
# Cythonize the modules

# Prepare by copying all .py modules to .pyx

from distutils.core import setup
from Cython.Build import cythonize

setup(name = 'Parallel GRASP (G)OP solver',
    ext_modules = cythonize("*.pyx"))

