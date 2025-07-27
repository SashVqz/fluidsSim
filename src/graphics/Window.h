#pragma once

#include <string>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

class Window {
public:
    Window(int width, int height, const std::string& title);
    ~Window();

    bool create();
    void makeCurrent();
    bool shouldClose() const;
    void setShouldClose(bool close);
    void swapBuffers();
    int getWidth() const;
    int getHeight() const;
    void setFramebufferSizeCallback(GLFWframebuffersizefun callback);

    GLFWwindow* getGlfwWindow() const { return glfwWindow; }
    bool isKeyPressed(int key) const;

private:
    GLFWwindow* glfwWindow;
    int width;
    int height;
    std::string title;
};