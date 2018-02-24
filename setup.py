from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name="Cython Fast Cube App",
    ext_modules=cythonize('move_step.pyx'),
    include_dirs=[numpy.get_include()],
)
