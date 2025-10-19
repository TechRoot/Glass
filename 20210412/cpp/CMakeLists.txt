cmake_minimum_required(VERSION 3.8)
project(pseudoinverse_lib LANGUAGES CXX)

# Biblioteca con funciones de pseudoinversa y número de condición
add_library(pseudoinverse STATIC
    src/pseudoinverse.cpp
)
target_include_directories(pseudoinverse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(pseudoinverse PUBLIC cxx_std_17)

# Ejecutable de pruebas
add_executable(test_pseudoinverse
    ../tests/cpp/test_pseudoinverse.cpp
    src/pseudoinverse.cpp
)
target_include_directories(test_pseudoinverse PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(test_pseudoinverse PRIVATE cxx_std_17)
target_link_libraries(test_pseudoinverse PRIVATE pseudoinverse)

# -----------------------------------------------------------------------------
# Biblioteca de grafos (algoritmo de Dijkstra)
# Se define una biblioteca separada con el algoritmo de Dijkstra y un ejecutable
# de pruebas asociado.  Se compilan con el estándar C++17.

# Biblioteca de grafos (Dijkstra)
add_library(graphs STATIC
    src/dijkstra.cpp
)
target_include_directories(graphs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(graphs PUBLIC cxx_std_17)

# Ejecutable de pruebas para Dijkstra
add_executable(test_dijkstra
    ../tests/cpp/test_dijkstra.cpp
    src/dijkstra.cpp
)
target_include_directories(test_dijkstra PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(test_dijkstra PRIVATE cxx_std_17)
target_link_libraries(test_dijkstra PRIVATE graphs)

# -----------------------------------------------------------------------------
# Módulo de interpolación y mínimos cuadrados (factorización QR)
#
# Se construye una biblioteca estática con las funciones de factorización QR.
# La implementación se encuentra principalmente en el encabezado `qr.hpp` y
# un archivo fuente vacío. Se proporciona un ejecutable de pruebas que
# verifica la descomposición y la resolución de sistemas lineales.

add_library(qr_lib STATIC
    src/qr.cpp
)
target_include_directories(qr_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qr_lib PUBLIC cxx_std_17)

add_executable(test_qr
    ../tests/cpp/test_qr.cpp
    src/qr.cpp
)
target_include_directories(test_qr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(test_qr PRIVATE cxx_std_17)
target_link_libraries(test_qr PRIVATE qr_lib)