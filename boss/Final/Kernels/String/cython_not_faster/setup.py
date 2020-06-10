from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


ext = Extension(
    "cy_string_kernel",
    sources=["cy_string_kernel.pyx"],
    language="c++",
    include_dirs=[np.get_include()]
)
setup(
    name="C String Kernel",
    ext_modules=cythonize(ext)
)

#install with python setup.py build_ext --inplace




