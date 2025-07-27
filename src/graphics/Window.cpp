#include "Window.h"
#include "utils/Logger.h"

// Constructor
Window::Window(int width, int height, const std::string& title)
    : glfwWindow(nullptr), width(width), height(height), title(title) {
}

// Destructor
Window::~Window() {
    if (glfwWindow) {
        glfwDestroyWindow(glfwWindow);
        glfwWindow = nullptr;
        Logger::info("Window '" + title + "' destroyed.");
    }
}

// Creation of the GLFW window
bool Window::create() {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // OpenGL 4.6
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6); // Changed to 4.6 for consistency with Application.cpp
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

// Make this window's OpenGL context current
void Window::makeCurrent() {
    if (glfwWindow) {
        glfwMakeContextCurrent(glfwWindow);
    }
}

// Check if the window should close
bool Window::shouldClose() const {
    return glfwWindow ? glfwWindowShouldClose(glfwWindow) : true;
}

// Set if the window should close
void Window::setShouldClose(bool close) {
    if (glfwWindow) {
        glfwSetWindowShouldClose(glfwWindow, close);
    }
}

// Swap front and back buffers
void Window::swapBuffers() {
    if (glfwWindow) {
        glfwSwapBuffers(glfwWindow);
    }
}

// Get current framebuffer width
int Window::getWidth() const {
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(glfwWindow, &fbWidth, &fbHeight);
    return fbWidth;
}

// Get current framebuffer height
int Window::getHeight() const {
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(glfwWindow, &fbWidth, &fbHeight);
    return fbHeight;
}

// Set framebuffer size callback
void Window::setFramebufferSizeCallback(GLFWframebuffersizefun callback) {
    if (glfwWindow) {
        glfwSetFramebufferSizeCallback(glfwWindow, callback);
    }
}

bool Window::isKeyPressed(int key) const {
    if (glfwWindow) {
        return glfwGetKey(glfwWindow, key) == GLFW_PRESS;
    }
    return false;
}
