cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DeployModel)

find_package(Torch REQUIRED)

add_executable(DeployModel main.cpp)
target_link_libraries(DeployModel "${TORCH_LIBRARIES}")
set_property(TARGET DeployModel PROPERTY CXX_STANDARD 11)
