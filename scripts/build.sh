#!/bin/bash

cd ..
mkdir -p build
mkdir -p bin

cd build
cmake ..
cmake --build . -- -j

cd ../scripts
