# Habilitar lenguaje CUDA
enable_language(CUDA)

set(PHYSICS_SOURCES
    FluidSimulator.h
    FluidSimulator.cpp
    SphFluidModel.h
    SphFluidModel.cpp
    sph_kernels.h
    sph_kernels.cu
)

add_library(Physics STATIC ${PHYSICS_SOURCES})

set_target_properties(Physics PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(Physics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/PhysX/physx/include  # PhysX  headers
)

target_link_libraries(Physics PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    glm 
)

target_compile_options(Physics PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
    >
)

set_target_properties(Physics PROPERTIES
    CUDA_ARCHITECTURES "50;60;70;75;80;86"
)

if(WIN32)
    set_target_properties(Physics PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()