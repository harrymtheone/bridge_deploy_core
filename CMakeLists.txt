cmake_minimum_required(VERSION 3.14)
project(bridge_core)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(BUILD_MUJOCO_INTERFACE "Build the MuJoCo simulation interface" ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(yaml-cpp REQUIRED)

# MuJoCo interface dependencies (optional)
if(BUILD_MUJOCO_INTERFACE)
    find_package(std_srvs REQUIRED)
    find_package(mujoco_ros_msgs REQUIRED)
    message(STATUS "MuJoCo interface: ENABLED")
else()
    message(STATUS "MuJoCo interface: DISABLED")
endif()

# Find ONNX Runtime manually (skip broken CMake config)
# Use ONNXRUNTIME_ROOT_DIR environment variable
set(ONNXRUNTIME_ROOT_DIR $ENV{ONNXRUNTIME_ROOT_DIR})
if(NOT ONNXRUNTIME_ROOT_DIR)
    set(ONNXRUNTIME_ROOT_DIR $ENV{ONNXRUNTIME_ROOT})
endif()

message(STATUS "Looking for ONNX Runtime in: ${ONNXRUNTIME_ROOT_DIR}")

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        ${ONNXRUNTIME_ROOT_DIR}/include
        ${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime
        /usr/local/include/onnxruntime
        /usr/include/onnxruntime
    NO_DEFAULT_PATH
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS
        ${ONNXRUNTIME_ROOT_DIR}/lib
        ${ONNXRUNTIME_ROOT_DIR}/lib64
        /usr/local/lib
        /usr/local/lib64
        /usr/lib
        /usr/lib64
    NO_DEFAULT_PATH
)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    message(STATUS "Found ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "Found ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
else()
    message(FATAL_ERROR "ONNX Runtime not found. Please set ONNXRUNTIME_ROOT_DIR environment variable.")
endif()

# Source files
set(BRIDGE_CORE_SOURCES
    src/core/config_manager.cpp
    src/core/state_machine.cpp
    src/core/rl_controller.cpp
    src/algorithms/algorithm_base.cpp
    src/algorithms/mod.cpp
    src/algorithms/dreamwaq.cpp
)

if(BUILD_MUJOCO_INTERFACE)
    list(APPEND BRIDGE_CORE_SOURCES src/interfaces/mujoco_interface.cpp)
endif()

# Create library
add_library(${PROJECT_NAME} SHARED ${BRIDGE_CORE_SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}
    ${ONNXRUNTIME_LIBRARY}
    yaml-cpp
)

# Base dependencies
set(BRIDGE_CORE_DEPS
    rclcpp
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf2_ros
)

if(BUILD_MUJOCO_INTERFACE)
    list(APPEND BRIDGE_CORE_DEPS std_srvs mujoco_ros_msgs)
endif()

ament_target_dependencies(${PROJECT_NAME} ${BRIDGE_CORE_DEPS})

# Install headers
install(
    DIRECTORY include/
    DESTINATION include
)

# Install library
install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Export targets
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

# Export package
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}Targets)
# Export dependencies
set(BRIDGE_CORE_EXPORT_DEPS
    rclcpp
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf2_ros
    yaml-cpp
)

if(BUILD_MUJOCO_INTERFACE)
    list(APPEND BRIDGE_CORE_EXPORT_DEPS std_srvs mujoco_ros_msgs)
endif()

ament_export_dependencies(${BRIDGE_CORE_EXPORT_DEPS})

ament_package()

