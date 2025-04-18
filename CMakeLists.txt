cmake_minimum_required (VERSION 3.15)

project("colgen_pricing" CXX)
find_package(highs REQUIRED)
find_path(QUILL_INCLUDE_DIRS "quill/Backend.h")

# Any source files added to include/ or source/ will automatically be added to the project.
file(GLOB_RECURSE INCLUDE_FILES src/*.h)
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)

include_directories(${QUILL_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} ${INCLUDE_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} highs::highs)
target_include_directories(${PROJECT_NAME} PUBLIC src)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)
