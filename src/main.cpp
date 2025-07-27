#include "core/Application.h"
#include "utils/Logger.h"

int main() {
    Logger::init("fluidsSim.log");
    Logger::info("Starting FluidsSim application");

    Application* app = new Application();

    if (!app->initialize()) {
        Logger::error("Application initialization failed!");
        delete app;
        Logger::shutdown();
        return -1;
    }

    app->run();
    Logger::info("Shutting down.");

    delete app;
    Logger::shutdown();
    
    return 0;
}