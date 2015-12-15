from setuptools import setup
from distutils.extension import Extension
import numpy

ext_modules = [
    Extension("emzed_optimizations.sample",
              ["emzed_optimizations/sample.c"],

              )
]

version = "0.4.2"

setup(name="emzed_optimizations",
      version=version,
      author="Uwe Schmitt",
      author_email="mail@uweschmitt.info",
      description="particular optimizations for speeding up emzed",
      license="BSD",
      url="http://github.com/uweschmitt/emzed_optimizations",
      packages=["emzed_optimizations"],
      zip_safe=False,
      include_dirs=[numpy.get_include()],
      ext_modules=ext_modules)
