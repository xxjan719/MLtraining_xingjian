cmake_minimum_required(VERSION 2.8)
project(demo)


include_directories(sub)

get_property(dirs DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
message(">>> inbclude_dirs=${dirs}")

add_executable(demo main.cpp)


