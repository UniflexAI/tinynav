cmake_minimum_required(VERSION 3.15)
project(tinynav_cpp_bind)

find_package(pybind11 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

pybind11_add_module(tinynav_cpp_bind
    cpp_bind_definitions.cpp
    raycast.cpp
    pose_graph_solver.cpp
    bundle_adjustment.cpp
)

target_include_directories(tinynav_cpp_bind PRIVATE
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(tinynav_cpp_bind PRIVATE ceres Eigen3::Eigen)
target_compile_features(tinynav_cpp_bind PRIVATE cxx_std_17)

install(TARGETS tinynav_cpp_bind DESTINATION .)
