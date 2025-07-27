
add_library(Physics STATIC
    FluidSimulator.h
    FluidSimulator.cpp
    SphFluidModel.h
    SphFluidModel.cpp
)

target_include_directories(Physics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
