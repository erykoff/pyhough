import os
from distutils.core import setup, Extension
import numpy

ext=Extension("pyhough._pyhough_pywrap", 
              ["pyhough/pyhough_pywrap.c","pyhough/pyhough.c"],
              extra_compile_args = ['-std=gnu99'])

setup(name="pyhough", 
      version="0.0.1",
      description="Hough transform in python",
      license = "GPL",
      author="Eli Rykoff",
      author_email="erykoff@gmail.com",
      ext_modules=[ext],
      include_dirs=numpy.get_include(),
      packages=['pyhough'])


