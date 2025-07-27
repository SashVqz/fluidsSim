cmake_minimum_required(VERSION 3.15)
project(fluidsSim LANGUAGES C CXX CUDA)

cmake_policy(SET CMP0104 NEW)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "50;60;70;75;80;86")
find_package(CUDAToolkit REQUIRED)

find_package(OpenGL REQUIRED)

set(GLFW_ROOT "${CMAKE_SOURCE_DIR}/external/glfw")
set(GLAD_ROOT "${CMAKE_SOURCE_DIR}/external/glad")
set(GLM_ROOT "${CMAKE_SOURCE_DIR}/external/glm")
set(PHYSX_ROOT "${CMAKE_SOURCE_DIR}/external/PhysX/physx")

# GLFW config
if(EXISTS "${GLFW_ROOT}/CMakeLists.txt")
    message(STATUS "Configuring GLFW from source...")
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory(${GLFW_ROOT})
    set(GLFW_FOUND TRUE)
    message(STATUS "GLFW configured successfully")
elseif(EXISTS "${GLFW_ROOT}/include/GLFW/glfw3.h")
    message(STATUS "Using precompiled GLFW...")

    set(GLFW_LIB_DIRS
        "${GLFW_ROOT}/lib-vc2022"
        "${GLFW_ROOT}/lib-static-ucrt"
        "${GLFW_ROOT}/lib-vc2019"
        "${GLFW_ROOT}/lib-vc2017"
    )

    set(GLFW_LIB_FOUND FALSE)
    foreach(LIB_DIR ${GLFW_LIB_DIRS})
        if(EXISTS "${LIB_DIR}/glfw3.lib")
            message(STATUS "Found GLFW library at: ${LIB_DIR}")
            add_library(glfw STATIC IMPORTED)
            set_target_properties(glfw PROPERTIES
                IMPORTED_LOCATION "${LIB_DIR}/glfw3.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${GLFW_ROOT}/include"
            )
            set(GLFW_FOUND TRUE)
            set(GLFW_LIB_FOUND TRUE)
            break()
        endif()
    endforeach()

    if(NOT GLFW_LIB_FOUND)
        message(FATAL_ERROR "GLFW library file (glfw3.lib) not found in any lib directory")
    endif()
else()
    message(FATAL_ERROR "GLFW not found at ${GLFW_ROOT}")
endif()

# GLAD config
if(EXISTS "${GLAD_ROOT}/src/glad.c" AND EXISTS "${GLAD_ROOT}/include/glad/glad.h")
    message(STATUS "Configuring GLAD...")
    add_library(glad STATIC ${GLAD_ROOT}/src/glad.c)
    target_include_directories(glad PUBLIC ${GLAD_ROOT}/include)

    set_target_properties(glad PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED ON
    )

    target_link_libraries(glad PRIVATE
        ucrt
        vcruntime
    )

    set_target_properties(glad PROPERTIES
        MSVC_RUNTIME_LIBRARY "$<$<CONFIG:Debug>:MultiThreadedDebugDLL>$<$<CONFIG:Release>:MultiThreadedDLL>")

    set(GLAD_FOUND TRUE)
    message(STATUS "GLAD configured successfully")
else()
    message(FATAL_ERROR "GLAD not found or incomplete at ${GLAD_ROOT}")
endif()

# GLM config
if(EXISTS "${GLM_ROOT}/glm/glm.hpp")
    message(STATUS "Configuring GLM...")
    add_library(glm INTERFACE)
    target_include_directories(glm INTERFACE ${GLM_ROOT})
    set(GLM_FOUND TRUE)
    message(STATUS "GLM configured successfully")
else()
    message(FATAL_ERROR "GLM not found at ${GLM_ROOT}")
endif()

# PhysX config
set(PHYSX_PLATFORM "win.x86_64.vc143.mt")
set(PHYSX_CONFIG "debug")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(PHYSX_CONFIG "release")
endif()

set(PHYSX_INCLUDE_DIRS "${PHYSX_ROOT}/include")
set(PHYSX_LIBRARY_DIRS "${PHYSX_ROOT}/bin/${PHYSX_PLATFORM}/${PHYSX_CONFIG}")

set(PHYSX_FOUND FALSE)
if(EXISTS "${PHYSX_LIBRARY_DIRS}/PhysX_64.dll")
    set(PHYSX_FOUND TRUE)
    message(STATUS "Found PhysX binaries at: ${PHYSX_LIBRARY_DIRS}")
else()
    set(PHYSX_CONFIG "release")
    set(PHYSX_LIBRARY_DIRS "${PHYSX_ROOT}/bin/${PHYSX_PLATFORM}/${PHYSX_CONFIG}")
    if(EXISTS "${PHYSX_LIBRARY_DIRS}/PhysX_64.dll")
        set(PHYSX_FOUND TRUE)
        message(STATUS "Found PhysX release binaries at: ${PHYSX_LIBRARY_DIRS}")
    else()
        file(GLOB_RECURSE PHYSX_DLLS "${PHYSX_ROOT}/bin/**/PhysX_64.dll")
        if(PHYSX_DLLS)
            list(GET PHYSX_DLLS 0 FIRST_DLL)
            get_filename_component(PHYSX_LIBRARY_DIRS "${FIRST_DLL}" DIRECTORY)
            set(PHYSX_FOUND TRUE)
            message(STATUS "Auto-detected PhysX binaries at: ${PHYSX_LIBRARY_DIRS}")
        endif()
    endif()
endif()

if(NOT PHYSX_FOUND)
    message(FATAL_ERROR "PhysX binaries not found. Please compile PhysX first!")
endif()

add_subdirectory(src/utils)
add_subdirectory(src/graphics)
add_subdirectory(src/physics)
add_subdirectory(src/core)

add_executable(fluidsSim src/main.cpp)

if(WIN32)
    set_target_properties(fluidsSim PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
        CUDA_ARCHITECTURES "50;60;70;75;80;86"
        # CUDA_SEPARABLE_COMPILATION ON
    )

    set_target_properties(Utils PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    
    set_target_properties(Graphics PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    
    set_target_properties(Physics PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
        CUDA_ARCHITECTURES "50;60;70;75;80;86"
        # CUDA_SEPARABLE_COMPILATION ON
    )
    
    set_target_properties(Core PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

target_link_libraries(fluidsSim PRIVATE
    Core
    Graphics
    Physics
    Utils
    OpenGL::GL
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cudart_static
    glfw
    glad
    glm
)

target_include_directories(fluidsSim PRIVATE ${PHYSX_INCLUDE_DIRS})
target_link_directories(fluidsSim PRIVATE ${PHYSX_LIBRARY_DIRS})

target_link_libraries(fluidsSim PRIVATE
    PhysX_64
    PhysXCommon_64
    PhysXFoundation_64
    PhysXExtensions_static_64
    PhysXPvdSDK_static_64
    PhysXCudaContextManager_static_64
)



if(WIN32)
    add_custom_command(TARGET fluidsSim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PHYSX_LIBRARY_DIRS}"
        "$<TARGET_FILE_DIR:fluidsSim>"
        COMMENT "Copying PhysX DLLs to output directory")
endif()

if(WIN32)
    set_target_properties(fluidsSim PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()

if(WIN32)
    set_target_properties(fluidsSim PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF)

    set_target_properties(Physics PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF)

    target_link_libraries(fluidsSim PRIVATE
        legacy_stdio_definitions
        ucrt
        vcruntime
    )

    set_target_properties(fluidsSim PROPERTIES
        MSVC_RUNTIME_LIBRARY "$<$<CONFIG:Debug>:MultiThreadedDebugDLL>$<$<CONFIG:Release>:MultiThreadedDLL>")

    set_target_properties(Physics PROPERTIES
        MSVC_RUNTIME_LIBRARY "$<$<CONFIG:Debug>:MultiThreadedDebugDLL>$<$<CONFIG:Release>:MultiThreadedDLL>")

    set_target_properties(Graphics PROPERTIES
        MSVC_RUNTIME_LIBRARY "$<$<CONFIG:Debug>:MultiThreadedDebugDLL>$<$<CONFIG:Release>:MultiThreadedDLL>")

    set_target_properties(Core PROPERTIES
        MSVC_RUNTIME_LIBRARY "$<$<CONFIG:Debug>:MultiThreadedDebugDLL>$<$<CONFIG:Release>:MultiThreadedDLL>")

    set_target_properties(Utils PROPERTIES
        MSVC_RUNTIME_LIBRARY "$<$<CONFIG:Debug>:MultiThreadedDebugDLL>$<$<CONFIG:Release>:MultiThreadedDLL>")
endif()

# check if all required packages are found
message(STATUS "Configuration Summary")
message(STATUS "GLFW:   ${GLFW_FOUND}")
message(STATUS "GLAD:   ${GLAD_FOUND}")
message(STATUS "GLM:    ${GLM_FOUND}")
message(STATUS "PhysX:  ${PHYSX_FOUND}")
message(STATUS "CUDA:   ${CUDAToolkit_FOUND}")
message(STATUS "OpenGL: ${OPENGL_FOUND}")