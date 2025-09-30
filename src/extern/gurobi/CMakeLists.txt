# Gurobi support
find_path(GUROBI_INCLUDE_DIRS
    NAMES gurobi_c.h
    HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
    PATH_SUFFIXES include)

if(GUROBI_INCLUDE_DIRS MATCHES "-NOTFOUND$")
    unset(GUROBI_INCLUDE_DIRS CACHE)
    unset(GUROBI_INCLUDE_DIRS)
    message(STATUS "Gurobi disabled: header not found")
else()
    find_library(GUROBI_LIB
        NAMES libgurobi120.so gurobi120.lib
        HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
        PATH_SUFFIXES lib)

    if(GUROBI_LIB MATCHES "-NOTFOUND$")
        unset(GUROBI_LIB CACHE)
        unset(GUROBI_LIB)
        message(STATUS "Gurobi disabled: library not found")
    else()
        set(GUROBI_ENABLED TRUE CACHE BOOL "Gurobi detected")
    endif()
endif()
