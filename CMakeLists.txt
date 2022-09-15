cmake_minimum_required(VERSION 3.7)

project(buTests VERSION 0.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

add_executable(randomTests
	tests/randomTests.cpp
	externals/bayesicUtilities/random.cpp
	)
add_compile_options(-Wall -Wextra -Wpedantic -march=native -fpic -O3)
