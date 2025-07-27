#pragma once

#include "graphics/Window.h"
#include "graphics/Camera.h"
#include "graphics/Shader.h"
#include "graphics/Mesh.h"
#include "physics/SphFluidModel.h"

#include <memory> 
#include <vector> 

class Application {
public:
    Application();
    ~Application();

    bool initialize();
    void run();
    void shutdown();

private:
    std::unique_ptr<Window> mainWindow;
    std::unique_ptr<Camera> mainCamera;

    // shaders 
    std::unique_ptr<Shader> basicShader;
    std::unique_ptr<Shader> particleShader; 

    // geometry
    std::unique_ptr<Mesh> containerMesh; 

    // SPH
    std::unique_ptr<SphFluidModel> fluidSimulator;

    unsigned int particleVAO;
    unsigned int particleVBO;
    float lastFrameTime;
    double lastMouseX;
    double lastMouseY;
    bool firstMouse;
    bool cameraEnabled;

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    void processInput(float deltaTime);
};