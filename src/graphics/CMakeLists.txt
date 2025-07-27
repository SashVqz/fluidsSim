add_library(Graphics STATIC
    Window.h
    Window.cpp
    Shader.h
    Shader.cpp
    Camera.h
    Camera.cpp
    Mesh.h
    Mesh.cpp
)

target_include_directories(Graphics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(Graphics PUBLIC
    glad
    glfw
    glm
    OpenGL::GL
)

if(WIN32)
    set_target_properties(Graphics PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()