#include "Logger.h"

// Inicialización de las variables miembro estáticas
std::ofstream Logger::s_logFile;
std::mutex Logger::s_logMutex;

// Inicializa el logger
void Logger::init(const std::string& logFileName) {
    std::lock_guard<std::mutex> lock(s_logMutex); // Bloquear para acceso seguro
    if (s_logFile.is_open()) {
        s_logFile.close();
    }
    s_logFile.open(logFileName, std::ios::out | std::ios::app); // Abrir en modo append
    if (s_logFile.is_open()) {
        std::cout << "Logger initialized. Messages will be written to " << logFileName << std::endl;
        s_logFile << "--- Logger Initialized at " << getTimestamp() << " ---" << std::endl;
    } else {
        std::cerr << "ERROR: Failed to open log file: " << logFileName << std::endl;
    }
}

// Apaga el logger
void Logger::shutdown() {
    std::lock_guard<std::mutex> lock(s_logMutex); // Bloquear para acceso seguro
    if (s_logFile.is_open()) {
        s_logFile << "--- Logger Shutting Down at " << getTimestamp() << " ---" << std::endl;
        s_logFile.close();
        std::cout << "Logger shut down. Log file closed." << std::endl;
    }
}

// Métodos públicos para registrar
void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warn(const std::string& message) {
    log(LogLevel::WARN, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

// Método privado para formatear y escribir el mensaje
void Logger::log(LogLevel level, const std::string& message) {
    std::lock_guard<std::mutex> lock(s_logMutex); // Asegurar acceso exclusivo al stream

    std::string timestamp = getTimestamp();
    std::string levelString = getLevelString(level);
    std::string formattedMessage = timestamp + " [" + levelString + "] " + message;

    // Imprimir en la consola
    if (level == LogLevel::ERROR) {
        std::cerr << formattedMessage << std::endl;
    } else {
        std::cout << formattedMessage << std::endl;
    }

    // Escribir en el archivo de log si está abierto
    if (s_logFile.is_open()) {
        s_logFile << formattedMessage << std::endl;
    }
}

// Método privado para obtener la marca de tiempo actual
std::string Logger::getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    // Usa gmtime para UTC o localtime para la zona horaria local
    std::tm bt = *std::localtime(&in_time_t); 

    std::ostringstream ss;
    ss << std::put_time(&bt, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// Método privado para obtener el string del nivel de log
const char* Logger::getLevelString(LogLevel level) {
    switch (level) {
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARN: return "WARN";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}