#pragma once

#include "graphics/Window.h"
#include "graphics/Camera.h"
#include "graphics/Shader.h"
#include "graphics/Mesh.h"
#include "physics/SphFluidModel.h" // Incluir la declaración adelantada si no hay dependencia circular

#include <memory> // Para std::unique_ptr
#include <vector> // Necesario para los datos de las partículas

class Application {
public:
    Application();
    ~Application();

    bool initialize();
    void run();
    void shutdown();

private:
    // --- Miembros de la ventana y cámara ---
    std::unique_ptr<Window> mainWindow;
    std::unique_ptr<Camera> mainCamera;

    // --- Shaders ---
    std::unique_ptr<Shader> basicShader;
    std::unique_ptr<Shader> particleShader; // ¡Nuevo! Shader para las partículas

    // --- Geometría ---
    std::unique_ptr<Mesh> containerMesh; // Malla del contenedor

    // --- Simulación SPH ---
    std::unique_ptr<SphFluidModel> fluidSimulator; // ¡Nuevo! El simulador de fluidos SPH

    // --- Renderizado de partículas ---
    // IDs de OpenGL para el Vertex Array Object y Vertex Buffer Object de las partículas
    unsigned int particleVAO; // ¡Nuevo! VAO para las partículas
    unsigned int particleVBO; // ¡Nuevo! VBO para las posiciones/colores de las partículas

    // --- Control de tiempo y entrada ---
    float lastFrameTime;
    double lastMouseX;
    double lastMouseY;
    bool firstMouse;
    bool cameraEnabled; // Controla si la cámara responde al ratón

    // --- Callbacks estáticos de GLFW ---
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    // --- Funciones de ayuda ---
    void processInput(float deltaTime);
};