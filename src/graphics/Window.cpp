#include "Window.h"
#include "utils/Logger.h"

Window::Window(int width, int height, const std::string& title) : glfwWindow(nullptr), width(width), height(height), title(title) {
}

Window::~Window() {
    if (glfwWindow) {
        glfwDestroyWindow(glfwWindow);
        glfwWindow = nullptr;
        Logger::info("Window '" + title + "' destroyed.");
    }
}

bool Window::create() {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif
    
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

    glfwWindow = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!glfwWindow) {
        Logger::error("Failed to create GLFW window.");
        return false;
    }

    Logger::info("Window '" + title + "' created (" + std::to_string(width) + "x" + std::to_string(height) + ").");
    return true;
}

void Window::makeCurrent() {
    if (glfwWindow) glfwMakeContextCurrent(glfwWindow);
}

bool Window::shouldClose() const {
    return glfwWindow ? glfwWindowShouldClose(glfwWindow) : true;
}

void Window::setShouldClose(bool close) {
    if (glfwWindow) glfwSetWindowShouldClose(glfwWindow, close);
}

void Window::swapBuffers() {
    if (glfwWindow) glfwSwapBuffers(glfwWindow);
}

int Window::getWidth() const {
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(glfwWindow, &fbWidth, &fbHeight);
    return fbWidth;
}

int Window::getHeight() const {
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(glfwWindow, &fbWidth, &fbHeight);
    return fbHeight;
}

void Window::setFramebufferSizeCallback(GLFWframebuffersizefun callback) {
    if (glfwWindow) glfwSetFramebufferSizeCallback(glfwWindow, callback);
}

bool Window::isKeyPressed(int key) const {
    if (glfwWindow) return glfwGetKey(glfwWindow, key) == GLFW_PRESS;
    return false;
}
