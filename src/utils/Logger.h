#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <chrono>   // Para obtener la hora actual
#include <iomanip>  // Para formatear la hora
#include <mutex>    // Para asegurar que el logger sea thread-safe

// Enumeración para los diferentes niveles de mensaje
enum class LogLevel {
    INFO,
    WARN,
    ERROR
};

class Logger {
public:
    // TODO: Constructor y Destructor (privados si es un Singleton o estático puro sin instancia)
    // El logger a menudo se implementa como una clase con métodos estáticos para facilitar el acceso global.
    // Opcional: Si necesitas un archivo de log, podrías tener una instancia singleton.

    // Método para inicializar el logger (ej. configurar archivo de log)
    // TODO: Si decides escribir a un archivo, abre el archivo aquí.
    static void init(const std::string& logFileName = "sashflow.log");

    // Método para apagar el logger (ej. cerrar archivo de log)
    // TODO: Si usas un archivo de log, ciérralo aquí.
    static void shutdown();

    // Métodos estáticos para registrar mensajes en diferentes niveles
    static void info(const std::string& message);
    static void warn(const std::string& message);
    static void error(const std::string& message);

private:
    // TODO: Si escribes a un archivo, necesitas un ofstream estático
    static std::ofstream s_logFile;
    // TODO: Mutex para asegurar el acceso concurrente seguro al archivo de log (thread-safety)
    static std::mutex s_logMutex;

    // Método privado para formatear y escribir el mensaje
    static void log(LogLevel level, const std::string& message);

    // Método privado para obtener la marca de tiempo actual formateada
    static std::string getTimestamp();

    // TODO: Opcional: Nombres de nivel de log para imprimir
    static const char* getLevelString(LogLevel level);
};