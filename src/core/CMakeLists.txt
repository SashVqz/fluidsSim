add_library(Core STATIC
    Application.h
    Application.cpp
)

target_include_directories(Core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src 
)

target_link_libraries(Core PUBLIC
    Graphics
    Physics
    Utils
    glad
    glfw
    glm
    OpenGL::GL
    CUDA::cudart
)

if(WIN32)
    set_target_properties(Core PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()