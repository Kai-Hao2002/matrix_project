cmake_minimum_required(VERSION 3.10)
project(MatrixMultiply)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 啟用 OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 找 OpenBLAS
find_package(OpenBLAS REQUIRED)

add_executable(matrix_multiply
    main.cpp
    gemm.cpp
    strassen.cpp
    strassen_multi_3x2x2.cpp
)

target_include_directories(matrix_multiply PRIVATE ${OpenBLAS_INCLUDE_DIRS})

target_link_libraries(matrix_multiply PRIVATE OpenBLAS::OpenBLAS OpenMP::OpenMP_CXX)

# 編譯最佳化（可視情況增減）
set_target_properties(matrix_multiply PROPERTIES
    COMPILE_FLAGS "-O3"
)
