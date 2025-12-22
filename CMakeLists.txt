cmake_minimum_required (VERSION 3.15)

project("colgen_pricing" CXX)
find_package(highs REQUIRED)
find_package(argparse CONFIG REQUIRED)
find_package(Taskflow CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(ZLIB REQUIRED)
find_path(QUILL_INCLUDE_DIRS "quill/Backend.h")

# Any source files added to include/ or source/ will automatically be added to the project.
file(GLOB_RECURSE INCLUDE_FILES src/*.h)
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)

# Search for Gurobi Support
add_subdirectory(src/extern/gurobi)

include_directories(${GUROBI_INCLUDE_DIRS})
include_directories(${QUILL_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} ${INCLUDE_FILES} ${SOURCE_FILES})

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

if(supported)
    message(STATUS "IPO / LTO enabled")
    set_property(TARGET colgen_pricing PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif()

if(GUROBI_ENABLED) 
    message(STATUS "Gurobi enabled")
    message(STATUS "Gurobi header found: ${GUROBI_INCLUDE_DIRS}")
    message(STATUS "Gurobi library found: ${GUROBI_LIB}")
    target_compile_definitions(${PROJECT_NAME} PRIVATE SUPPORT_GUROBI) 
endif()

target_link_libraries(${PROJECT_NAME} highs::highs ${GUROBI_LIB} nlohmann_json::nlohmann_json ZLIB::ZLIB Eigen3::Eigen)
target_include_directories(${PROJECT_NAME} PUBLIC src)


set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)
