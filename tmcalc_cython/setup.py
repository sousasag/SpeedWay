from distutils.core import setup, Extension
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("tmcalc_module", ["tmcalc_module.pyx"])]
setup(cmdclass={'build_ext': build_ext}, include_dirs = [np.get_include()], ext_modules= ext_modules)

