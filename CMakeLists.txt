cmake_minimum_required(VERSION 3.18)
project(hedgelab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# Core library: plain C++, no Python dependency.
add_library(hedgelab STATIC
  cpp/src/black_scholes.cpp
  cpp/src/binomial.cpp
  cpp/src/monte_carlo.cpp
  cpp/src/hedging.cpp)
target_include_directories(hedgelab PUBLIC cpp/include)
target_link_libraries(hedgelab PUBLIC Threads::Threads)
target_compile_options(hedgelab PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(hedgelab PROPERTIES POSITION_INDEPENDENT_CODE ON)

# C++ unit tests.
enable_testing()
add_executable(test_hedgelab cpp/tests/test_core.cpp)
target_link_libraries(test_hedgelab PRIVATE hedgelab)
add_test(NAME hedgelab_tests COMMAND test_hedgelab)

# Python bindings: builds hedgelab/_core, placed next to the Python sources.
option(HEDGELAB_PYTHON "Build the pybind11 module hedgelab._core" ON)
if(HEDGELAB_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core cpp/bindings/module.cpp)
  target_link_libraries(_core PRIVATE hedgelab)
  # $<1:...> stops multi-config generators from appending a Debug/Release subfolder.
  set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_SOURCE_DIR}/src/hedgelab>")
endif()
