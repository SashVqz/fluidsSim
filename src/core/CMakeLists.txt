# Define una librería estática llamada 'Core'
# Esta librería incluirá los archivos de código fuente Application.cpp
add_library(Core STATIC
    Application.h
    Application.cpp
)

# TODO: Define los directorios de inclusión públicos para la librería Core.
# Esto asegura que otros módulos (como main.cpp) puedan encontrar los archivos de cabecera de Core.
# 'PUBLIC' significa que los directorios de inclusión serán propagados a los objetivos que enlacen con 'Core'.
target_include_directories(Core PUBLIC
    # La ruta a este propio directorio de Core (donde está Application.h)
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# TODO: Si la librería Core necesitara enlazar directamente con alguna librería externa
# (aunque en este caso las dependencias están gestionadas en el CMakeLists.txt principal
# y se propagarán a través de la vinculación de Core al ejecutable principal),
# lo harías aquí. Por ejemplo, si Application usara directamente alguna función de GLFW
# que no fuera a través de Window.h/cpp, se podría añadir:
# target_link_libraries(Core PRIVATE GLFW::GLFW) # Ejemplo