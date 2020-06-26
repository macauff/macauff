from setuptools import find_packages
from numpy.distutils.core import Extension, setup

extension = []

setup(name="pytestf90", packages=find_packages(),
      ext_modules=extension)
