# Define una librería estática llamada 'Utils'
# Esta librería incluirá los archivos de código fuente del Logger.
add_library(Utils STATIC
    Logger.h
    Logger.cpp
)

# Define los directorios de inclusión públicos para la librería Utils.
# Esto asegura que otros módulos (como Application, Window, etc.) puedan
# encontrar los archivos de cabecera de Utils.
target_include_directories(Utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR} # Para que se encuentre Logger.h
)

# El Logger no tiene dependencias directas de otras librerías externas
# más allá de las bibliotecas estándar de C++, por lo que no se requiere
# target_link_libraries adicional aquí.