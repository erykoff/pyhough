import os
from distutils.core import setup, Extension
import numpy
import shutil
import glob


sources = ["pyhough/pyhough_pywrap.c","pyhough/pyhough.c"]


ext=Extension("pyhough._pyhough_pywrap",
              sources,
              extra_compile_args = ['-std=gnu99'])


setup(name="pyhough", 
      version="0.1",
      description="Hough transform in python",
      license = "GPL",
      author="Eli Rykoff",
      author_email="erykoff@gmail.com",
      ext_modules=[ext],
      include_dirs=[numpy.get_include()],
      packages=['pyhough'])


