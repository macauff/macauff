from setuptools import find_packages
from numpy.distutils.core import Extension, setup

extension = [Extension(name='macauff.perturbation_auf_fortran',
                       sources=['macauff/perturbation_auf_fortran.f90'], language='f90',
                       extra_link_args=["-lgomp"],
                       extra_f90_compile_args=["-Wall", "-Wextra", "-Werror", "-pedantic",
                                               "-fbacktrace", "-O0", "-g", "-fcheck=all",
                                               "-fopenmp"])]

setup(name="macauff", packages=find_packages(),
      package_data={'macauff': ['tests/data/*']},
      ext_modules=extension)
