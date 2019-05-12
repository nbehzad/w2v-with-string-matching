cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(string_match_w2v LANGUAGES C CUDA)

set(CMAKE_C_FLAGS "-g -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result")
#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY		${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG	${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE	${PROJECT_SOURCE_DIR}/bin)

file(GLOB C_SOURCES
	src/srlalgos/*.c
	src/word2vec.c
)

file(GLOB SERIAL_SOURCES
	src/srlalgos/*.c
)

file(GLOB CUDA_SOURCES
	src/util/*.cu
	src/wrappers/*.cu
	src/algos/*.cu
)

include_directories(${PROJECT_SOURCE_DIR}/src)

add_library(cpu-serial STATIC ${SERIAL_SOURCES})
add_library(gpu-parallel STATIC ${CUDA_SOURCES})

set_target_properties(gpu-parallel PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	)

add_executable(string_match_w2v ${C_SOURCES})
#target_link_libraries(string_match_w2v m utils)

target_link_libraries(string_match_w2v 
	PUBLIC m 
	PRIVATE cpu-serial 
	PRIVATE gpu-parallel
	)





