#include "core/Application.h"
#include "utils/Logger.h"

int main() {
    // Inicializar el sistema de logging al principio de la aplicación.
    // Todos los mensajes de log irán a la consola y a 'sashflow.log'.
    Logger::init("sashflow.log");

    Logger::info("Starting SashFlow application...");

    // Crear una instancia de la aplicación principal
    Application* app = new Application();

    // Inicializar la aplicación
    if (!app->initialize()) {
        Logger::error("Application initialization failed! Exiting.");
        delete app; // Liberar memoria antes de salir
        Logger::shutdown(); // Apagar el logger
        return -1; // Retornar un código de error
    }

    // Ejecutar el bucle principal de la aplicación
    app->run();

    // La aplicación ha terminado, realizar la limpieza
    Logger::info("SashFlow application finished. Shutting down.");

    delete app; // Liberar la memoria de la instancia de la aplicación

    // Apagar el sistema de logging al final
    Logger::shutdown();

    return 0; // Salida exitosa
}