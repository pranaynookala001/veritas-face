#!/usr/bin/env sh
set -eu

script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build_directory="$script_directory/build"

cmake -S "$script_directory" -B "$build_directory"
cmake --build "$build_directory"
ctest --test-dir "$build_directory" --output-on-failure
