# Define una librería estática llamada 'Graphics'
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

# TODO: Define los directorios de inclusión públicos para la librería Graphics.
# Esto es crucial para que otros módulos (como Application) puedan encontrar
# los archivos de cabecera de Graphics.
target_include_directories(Graphics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR} # Para que se encuentren Window.h, Shader.h, etc.
)

# TODO: Enlazar las librerías externas de las que depende Graphics.
# Graphics necesita GLAD, GLFW y GLM.
# Usaremos las dependencias que se configuran en el CMakeLists.txt principal
# y se propagan al enlazarse con el ejecutable.
# Sin embargo, si quieres que 'Graphics' tenga sus propias dependencias internas
# bien definidas, puedes hacerlo aquí.

# GLM es una librería solo de encabezados, no necesita un enlace directo aquí,
# solo la inclusión global que ya pusimos en el CMakeLists.txt principal.

# GLFW y GLAD son enlazados por el ejecutable principal.
# Si tuvieras que enlazar algo aquí directamente, sería así:
# target_link_libraries(Graphics PRIVATE
#     # Si Window.cpp necesita un enlace directo a GLFW:
#     # ${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/lib-vc2022/glfw3.lib
#     # Esto es manejado por el CMakeLists.txt principal que vincula GLFW directamente
#     # al objetivo ejecutable final que consume Graphics.
# )