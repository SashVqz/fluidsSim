#include "Logger.h"
#include <sstream>

std::ofstream Logger::s_logFile;
std::mutex Logger::s_logMutex;

void Logger::init(const std::string& logFileName) {
    std::lock_guard<std::mutex> lock(s_logMutex);
    if (s_logFile.is_open()) {
        s_logFile.close();
    }
    s_logFile.open(logFileName, std::ios::out | std::ios::app);
    if (s_logFile.is_open()) {
        std::cout << "Logger initialized. Messages will be written to " << logFileName << std::endl;
        s_logFile << "Logger Initialized at " << getTimestamp() << std::endl;
    } else {
        std::cerr << "ERROR: Failed to open log file: " << logFileName << std::endl;
    }
}

void Logger::shutdown() {
    std::lock_guard<std::mutex> lock(s_logMutex);
    if (s_logFile.is_open()) {
        s_logFile << "Logger Shutting Down at " << getTimestamp() << std::endl;
        s_logFile.close();
        std::cout << "Logger shut down. Log file closed." << std::endl;
    }
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warn(const std::string& message) {
    log(LogLevel::WARN, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

void Logger::log(LogLevel level, const std::string& message) {
    std::lock_guard<std::mutex> lock(s_logMutex);

    std::string timestamp = getTimestamp();
    std::string levelString = getLevelString(level);
    std::string formattedMessage = timestamp + " [" + levelString + "] " + message;

    if (level == LogLevel::ERROR) {
        std::cerr << formattedMessage << std::endl;
    } else {
        std::cout << formattedMessage << std::endl;
    }

    if (s_logFile.is_open()) s_logFile << formattedMessage << std::endl;
}

std::string Logger::getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm bt = *std::localtime(&in_time_t); 
    std::ostringstream ss;
    ss << std::put_time(&bt, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

const char* Logger::getLevelString(LogLevel level) {
    switch (level) {
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARN: return "WARN";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}