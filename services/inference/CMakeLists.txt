cmake_minimum_required(VERSION 3.25)
project(veritas_face_inference VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(VERITAS_ONNXRUNTIME_VERSION "1.20.1")
set(ONNXRUNTIME_ROOT "" CACHE PATH
    "Path to an extracted official ONNX Runtime C/C++ CPU package")
option(VERITAS_FETCH_ONNXRUNTIME
       "Download the pinned official ONNX Runtime C/C++ CPU package when ONNXRUNTIME_ROOT is unset"
       ON)

if(ONNXRUNTIME_ROOT)
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            HINTS "${ONNXRUNTIME_ROOT}/include"
            NO_DEFAULT_PATH
            REQUIRED)
  if(APPLE)
    file(GLOB ONNXRUNTIME_LIBRARY_CANDIDATES
         "${ONNXRUNTIME_ROOT}/lib/libonnxruntime*.dylib")
  else()
    file(GLOB ONNXRUNTIME_LIBRARY_CANDIDATES
         "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so*")
  endif()
  list(LENGTH ONNXRUNTIME_LIBRARY_CANDIDATES ONNXRUNTIME_LIBRARY_COUNT)
  if(ONNXRUNTIME_LIBRARY_COUNT EQUAL 0)
    message(FATAL_ERROR "The configured ONNX Runtime package has no shared library.")
  endif()
  list(GET ONNXRUNTIME_LIBRARY_CANDIDATES 0 ONNXRUNTIME_LIBRARY)
elseif(VERITAS_FETCH_ONNXRUNTIME)
  include(FetchContent)
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # The universal package avoids the known 1.20.1 arm64-only package loader issue.
    set(VERITAS_ONNXRUNTIME_ARCHIVE
        "onnxruntime-osx-universal2-${VERITAS_ONNXRUNTIME_VERSION}.tgz")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
         (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64"))
    set(VERITAS_ONNXRUNTIME_ARCHIVE
        "onnxruntime-linux-x64-${VERITAS_ONNXRUNTIME_VERSION}.tgz")
  else()
    message(FATAL_ERROR
            "Set ONNXRUNTIME_ROOT to an official CPU package for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}.")
  endif()

  FetchContent_Declare(
    veritas_onnxruntime
    URL "https://github.com/microsoft/onnxruntime/releases/download/v${VERITAS_ONNXRUNTIME_VERSION}/${VERITAS_ONNXRUNTIME_ARCHIVE}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    # The archive is an SDK rather than a CMake project. This prevents
    # FetchContent_MakeAvailable from trying to add a nonexistent subproject.
    SOURCE_SUBDIR "veritas-onnxruntime-sdk")
  FetchContent_MakeAvailable(veritas_onnxruntime)
  set(ONNXRUNTIME_INCLUDE_DIR "${veritas_onnxruntime_SOURCE_DIR}/include")
  if(APPLE)
    file(GLOB ONNXRUNTIME_LIBRARY_CANDIDATES
         "${veritas_onnxruntime_SOURCE_DIR}/lib/libonnxruntime*.dylib")
  else()
    file(GLOB ONNXRUNTIME_LIBRARY_CANDIDATES
         "${veritas_onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so*")
  endif()
  list(LENGTH ONNXRUNTIME_LIBRARY_CANDIDATES ONNXRUNTIME_LIBRARY_COUNT)
  if(ONNXRUNTIME_LIBRARY_COUNT EQUAL 0)
    message(FATAL_ERROR "The downloaded ONNX Runtime package has no shared library.")
  endif()
  list(GET ONNXRUNTIME_LIBRARY_CANDIDATES 0 ONNXRUNTIME_LIBRARY)
else()
  message(FATAL_ERROR "Set ONNXRUNTIME_ROOT or enable VERITAS_FETCH_ONNXRUNTIME.")
endif()

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h" OR
   NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
  message(FATAL_ERROR "The configured ONNX Runtime package is incomplete.")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")

add_executable(veritas-inference
  src/main.cpp
  src/face_preprocessor.cpp)
target_compile_options(veritas-inference PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(veritas-inference PRIVATE onnxruntime::onnxruntime)
get_filename_component(ONNXRUNTIME_LIBRARY_DIRECTORY "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
set_target_properties(veritas-inference PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIRECTORY}")

include(CTest)

if(BUILD_TESTING)
  find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)
  add_test(
    NAME inference-http-contract
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_http_contract.py
            $<TARGET_FILE:veritas-inference>
  )
  add_test(
    NAME inference-preprocessing-parity
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_preprocessing_parity.py
            $<TARGET_FILE:veritas-inference>
  )
endif()

# ONNXRUNTIME_ROOT makes offline/reproducible builds possible. The default fetch
# uses the pinned official CPU package for the local macOS and CI Linux targets.
