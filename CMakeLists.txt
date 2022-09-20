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

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -pedanitc")

add_executable(randomTests
	tests/randomTests.cpp
	externals/bayesicUtilities/random.cpp
)

target_compile_options(randomTests
	PRIVATE -mrdrnd
)

set_target_properties(randomTests PROPERTIES
	CXX_STANDARD         14
	CXX_STANDARD_REQURED ON
	CXX_EXTENSIONS       OFF
)

