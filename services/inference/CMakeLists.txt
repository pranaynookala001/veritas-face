cmake_minimum_required(VERSION 3.25)
project(veritas_face_inference VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(veritas-inference src/main.cpp)
target_compile_options(veritas-inference PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)

if(BUILD_TESTING)
  find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)
  add_test(
    NAME inference-http-contract
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_http_contract.py
            $<TARGET_FILE:veritas-inference>
  )
endif()

# The HTTP contract intentionally has no inference-runtime dependency yet. The
# following milestone supplies the ONNX implementation behind this stable API.
