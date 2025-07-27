#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <mutex> 

enum class LogLevel {
    INFO,
    WARN,
    ERROR
};

class Logger {
public:
    static void init(const std::string& logFileName = "fluidsSim.log");
    static void shutdown();
    static void info(const std::string& message);
    static void warn(const std::string& message);
    static void error(const std::string& message);

private:
    static std::ofstream s_logFile;
    static std::mutex s_logMutex;

    static void log(LogLevel level, const std::string& message);
    static std::string getTimestamp();
    static const char* getLevelString(LogLevel level);
};