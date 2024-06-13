cmake_minimum_required(VERSION 3.16)
project(2D_Quadrangulation)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)




option(LIBIGL_GLFW                "Build target igl::glfw"                ON)
option(LIBIGL_IMGUI               "Build target igl::imgui"               ON)
option(LIBIGL_OPENGL              "Build target igl::opengl"              ON)

# Libigl
include(libigl)

# # Enable the target igl::glfw
# igl_include(glfw)

# include directories
include_directories(${PROJECT_NAME} PRIVATE ./include)

# Add your project files
file(GLOB SRC_FILES src/*.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})

# linking libraries
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw igl::imgui igl::opengl)
