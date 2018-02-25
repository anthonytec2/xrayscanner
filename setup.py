from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


extensions = [ Extension("move_step", ["move_step.pyx"], extra_compile_args=["-O3"," -march=native"],extra_link_args=["-O3"," -march=native"])]

setup(
    name="Cython Fast Cube App",
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
