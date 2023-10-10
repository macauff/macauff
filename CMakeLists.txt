cmake_minimum_required(VERSION 3.18)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

find_package(OpenMP)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)

# F2PY headers
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c
          "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

add_library(fortranobject OBJECT "${F2PY_INCLUDE_DIR}/fortranobject.c")
target_link_libraries(fortranobject PUBLIC Python::NumPy)
target_include_directories(fortranobject PUBLIC "${F2PY_INCLUDE_DIR}")
set_property(TARGET fortranobject PROPERTY POSITION_INDEPENDENT_CODE ON)

function(add_f2py_module module_name)
  add_custom_command(
    OUTPUT "${module_name}module.c" "${module_name}-f2pywrappers2.f90"
    DEPENDS "macauff/${module_name}.f90" "macauff/shared_library.f90"
    VERBATIM
    COMMAND "${Python_EXECUTABLE}" -m numpy.f2py
            "${CMAKE_CURRENT_SOURCE_DIR}/macauff/${module_name}.f90" -m "${module_name}")

  python_add_library("${module_name}" MODULE
                     "${CMAKE_CURRENT_BINARY_DIR}/${module_name}module.c"
                     "${CMAKE_CURRENT_BINARY_DIR}/${module_name}-f2pywrappers2.f90"
                     "${CMAKE_CURRENT_SOURCE_DIR}/macauff/${module_name}.f90"
                     "${CMAKE_CURRENT_SOURCE_DIR}/macauff/shared_library.f90"
                     WITH_SOABI)
  target_link_libraries("${module_name}" PRIVATE fortranobject)
  if(OpenMP_Fortran_FOUND)
    target_link_libraries("${module_name}" PRIVATE OpenMP::OpenMP_Fortran)
  endif()
  install(TARGETS "${module_name}" DESTINATION ${SKBUILD_PROJECT_NAME})
endfunction()

add_f2py_module("counterpart_pairing_fortran")
add_f2py_module("group_sources_fortran")
add_f2py_module("misc_functions_fortran")
add_f2py_module("perturbation_auf_fortran")
add_f2py_module("photometric_likelihood_fortran")
