import os
from setuptools import setup, Extension
import numpy
import shutil
import glob


sources = ["pyhough/pyhough_pywrap.c","pyhough/pyhough.c"]

sources2 = ["pyhough/pyhoughback_pywrap.c","pyhough/pyhoughback.c"]

ext = Extension("pyhough._pyhough_pywrap",
                sources,
                extra_compile_args = ['-std=gnu99'])

ext2 = Extension("pyhough._pyhoughback_pywrap",
                 sources2,
                 extra_compile_args = ['-std=gnu99'])

setup(name="pyhough",
      version="0.2",
      description="Hough transform in python",
      license = "GPL",
      author="Eli Rykoff",
      author_email="erykoff@gmail.com",
      ext_modules=[ext, ext2],
      include_dirs=[numpy.get_include()],
      packages=['pyhough'])


