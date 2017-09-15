cmake_minimum_required(VERSION 3.1)

project(Daydream)

# Check the OS -----------------------------------------------------------------
if(NOT UNIX)
	message(FATAL_ERROR " Only Unix based systems are supported")
endif()

# Check the CMake usage --------------------------------------------------------
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR " Inappropriate CMake usage. \n" 
	                    " Should be called in a build directory: \n" 
	                    " $ mkdir build; cmake ..")
endif()

# Find Eigen3 ------------------------------------------------------------------
find_package(Eigen3)
if(NOT EIGEN3_FOUND)
	message(FATAL_ERROR " Could not locate the Eigen3 library. \n"
	                    " Please ensure it is installed on your machine.")
endif()

# Include Daydream and Eigen3 --------------------------------------------------
include_directories(. ${EIGEN3_INCLUDE_DIRS})

# Set the compiler flags -------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O2 -s -Wall -Wextra")
endif()

# Build the sources ------------------------------------------------------------
file(GLOB_RECURSE DAYDREAM_SRC "src/*.cpp")
add_library(Daydream SHARED ${DAYDREAM_SRC})

# Install the headers and the compiled library ---------------------------------
install(DIRECTORY Daydream DESTINATION include)
install(TARGETS Daydream DESTINATION lib)

# Build the example ------------------------------------------------------------
#add_subdirectory(example)
option(BUILD_EXAMPLE "Build the example." ON)
if(BUILD_EXAMPLE)
	find_program(GNUPLOT_EXECUTABLE gnuplot)
	if (NOT GNUPLOT_EXECUTABLE)
		message(WARNING "GNUPlot is required to build the example.")
	else()
		add_executable(EXAMPLE example/example.cpp)
		target_link_libraries(EXAMPLE Daydream)
		execute_process(COMMAND ${CMAKE_SOURCE_DIR}/data/get_mnist.py
		                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/data)
	endif()
endif()

# TODO: subfolder for the example, the python script with image? NO
# TODO: Doxygen?
