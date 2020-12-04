from setuptools import find_packages
from numpy.distutils.core import Extension, setup

names = ['group_sources_fortran', 'misc_functions_fortran']
f90_args = ["-Wall", "-Wextra", "-Werror", "-pedantic", "-fbacktrace", "-O0", "-g", "-fcheck=all",
            "-fopenmp"]

extension = [Extension(name='macauff.{}'.format(name), sources=['macauff/{}.f90'.format(name)],
                       language='f90', extra_link_args=["-lgomp"],
                       extra_f90_compile_args=f90_args, libraries=['shared_library'])
             for name in names]

setup(name="macauff", packages=find_packages(), package_data={'macauff': ['tests/data/*']},
      ext_modules=extension, libraries=[('shared_library',
                                        dict(sources=['macauff/shared_library.f90'],
                                         extra_f90_compile_args=f90_args,
                                         extra_link_args=["-lgomp"], language="f90"))])
