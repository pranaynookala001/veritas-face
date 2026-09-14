cmake_minimum_required(VERSION 3.25)
project(veritas_face_inference VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(veritas-inference src/main.cpp)
target_compile_options(veritas-inference PRIVATE -Wall -Wextra -Wpedantic)

# Milestone 2 links ONNX Runtime and a minimal HTTP transport here. Keeping the
# scaffold dependency-free makes the initial C++ contract buildable everywhere.
