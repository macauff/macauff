from setuptools import find_packages
from numpy.distutils.core import Extension, setup

extension = []

setup(name="macauff", packages=find_packages(),
      package_data={'macauff': ['tests/data/*']},
      ext_modules=extension)
