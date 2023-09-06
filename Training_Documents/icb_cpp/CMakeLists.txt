cmake_minimum_required(VERSION 3.10)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Specify the required ARPACK and MPI libraries
find_package(ARPACKNG REQUIRED)
find_package(MPI REQUIRED)

# PARPACK Project
set(PARPACK_SOURCES "icb_parpack_cpp.cpp")
add_executable(icb_parpack_cpp ${PARPACK_SOURCES})
target_include_directories(icb_parpack_cpp INTERFACE PARPACK::PARPACK MPI::MPI_CXX)
target_link_libraries(icb_parpack_cpp PRIVATE PARPACK::PARPACK MPI::MPI_CXX)

# ARPACK Project
set(ARPACK_SOURCES "icb_arpack_cpp.cpp")
add_executable(icb_arpack_cpp ${ARPACK_SOURCES})
target_include_directories(icb_arpack_cpp INTERFACE ARPACK::ARPACK)
target_link_libraries(icb_arpack_cpp PRIVATE ARPACK::ARPACK)
