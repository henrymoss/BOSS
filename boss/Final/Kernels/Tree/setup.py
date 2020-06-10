from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


ext = Extension(
    "C_tree_kernel",
    sources=["C_tree_kernel.pyx"],
    language="c++",
    include_dirs=[np.get_include()]
)
setup(
    name="C Tree Kernel",
    ext_modules=cythonize(ext)
)

#install with python setup.py build_ext --inplace




