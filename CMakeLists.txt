cmake_minimum_required(VERSION 3.16)
project(MultiHeadAttention)

set(CMAKE_CXX_STANDARD 20)
include_directories("model" "optimized")

add_executable(origin main.cpp)
add_executable(new new.cpp)