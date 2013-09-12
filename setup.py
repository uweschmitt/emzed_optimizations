from setuptools import setup
from distutils.extension import Extension

ext_modules = [
    Extension("emzed_optimizations.sample",
              ["emzed_optimizations/sample.c"],
              )
]

setup(name="emzed_optimizations",
      version="0.0.1",
      author="Uwe Schmitt",
      author_email="mail@uweschmitt.info",
      description="particular optimizations for speeding up emzed",
      license="BSD",
      url="http://github.com/uweschmitt/emzed_optimizations",
      packages=["emzed_optimizations"],
      zip_safe=False,
      ext_modules=ext_modules)
