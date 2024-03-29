cmake_minimum_required(VERSION 3.11)
project(buTests VERSION 0.1 LANGUAGES CXX)

# Adding a Profile build type
get_property(isMultiConfig GLOBAL
	PROPERTY GENERATOR_IS_MULTICONFIG
)
if(isMultiConfig)
	if(NOT "Profile" IN_LIST CMAKE_CONFIGURATION_TYPES)
		list(APPEND CMAKE_CONFIGURATION_TYPES Profile)
	endif()
else()
	set(allowedBuildTypes Debug Release Profile)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
		STRINGS "${allowedBuildTypes}"
	)
	if(NOT CMAKE_BUILD_TYPE)
		set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE) # default build type is Release
	elseif(NOT CMAKE_BUILD_TYPE IN_LIST allowedBuildTypes)
		message(FATAL_ERROR
			"Unknown build type ${CMAKE_BUILD_TYPE}"
		)
	endif()
endif()

# -mbmi2 is for PEXT and PDEP; -mbmi required for _tzcnt_u64()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion -Wpedantic -Wshadow -mbmi2 -mlzcnt -mbmi")

# tests of the RNG
add_executable(randomTests
	tests/randomTests.cpp
	externals/bayesicUtilities/src/random.cpp
)
target_include_directories(randomTests
	PRIVATE externals/bayesicUtilities/include
)

target_compile_options(randomTests
	PRIVATE -mrdrnd
)

set_target_properties(randomTests PROPERTIES
	CXX_STANDARD         14
	CXX_STANDARD_REQURED ON
	CXX_EXTENSIONS       OFF
)

# tests of the OPH function
add_executable(hashTests
	tests/hashTests.cpp
	externals/bayesicUtilities/src/random.cpp
)
target_include_directories(hashTests
	PRIVATE externals/bayesicUtilities/include
)

add_executable(ophCorrect
	tests/ophCorrectnessTest.cpp
	externals/bayesicUtilities/src/random.cpp
)
target_include_directories(ophCorrect
	PRIVATE externals/bayesicUtilities/include
)
set_target_properties(ophCorrect PROPERTIES
	CXX_STANDARD         14
	CXX_STANDARD_REQURED ON
	CXX_EXTENSIONS       OFF
)
# POPCNT tests
add_executable(popcnt
	tests/popcntTest.cpp
	externals/bayesicUtilities/src/random.cpp
)
target_include_directories(popcnt
	PRIVATE externals/bayesicUtilities/include
)
target_compile_options(popcnt
	PRIVATE -mpopcnt -mavx2 -msse2
)
set_target_properties(popcnt PROPERTIES
	CXX_STANDARD         14
	CXX_STANDARD_REQURED ON
	CXX_EXTENSIONS       OFF
)

# Fisher-Yates tests
add_executable(fyTests
	tests/fisherYatesTest.cpp
	externals/bayesicUtilities/src/random.cpp
)
target_include_directories(fyTests
	PRIVATE externals/bayesicUtilities/include
)

set_target_properties(fyTests PROPERTIES
	CXX_STANDARD         14
	CXX_STANDARD_REQURED ON
	CXX_EXTENSIONS       OFF
)

# PRNG tests
add_executable(prngTst
	tests/prngTests.cpp
	externals/bayesicUtilities/src/random.cpp
)
target_include_directories(prngTst
	PRIVATE externals/bayesicUtilities/include
)

set_target_properties(prngTst PROPERTIES
	CXX_STANDARD         14
	CXX_STANDARD_REQURED ON
	CXX_EXTENSIONS       OFF
)

# MurMurHash3 tests
add_executable(mmhTest
	tests/mmhTests.cpp
	externals/bayesicUtilities/src/random.cpp
)
target_include_directories(mmhTest
	PRIVATE externals/bayesicUtilities/include
)

set_target_properties(mmhTest PROPERTIES
	CXX_STANDARD         14
	CXX_STANDARD_REQURED ON
	CXX_EXTENSIONS       OFF
)
