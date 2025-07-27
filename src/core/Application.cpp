#include "Application.h"
#include "utils/Logger.h"
#include "graphics/Window.h"
#include "graphics/Shader.h"
#include "graphics/Camera.h"
#include "graphics/Mesh.h" 
#include "physics/SphFluidModel.h"

#include <glad/glad.h>
#include <gLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector> 

void Application::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app && app->mainCamera) {
        app->mainCamera->setAspectRatio((float)width / (float)height);
        Logger::info("Framebuffer resized to: " + std::to_string(width) + "x" + std::to_string(height));
    }
}

void Application::mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app || !app->cameraEnabled) return;

    if (app->firstMouse) {
        app->lastMouseX = static_cast<float>(xpos);
        app->lastMouseY = static_cast<float>(ypos);
        app->firstMouse = false;
    }

    float xoffset = static_cast<float>(xpos - app->lastMouseX);
    float yoffset = static_cast<float>(app->lastMouseY - ypos); // Invert y-axis for correct mouse movement

    app->lastMouseX = static_cast<float>(xpos);
    app->lastMouseY = static_cast<float>(ypos);

    app->mainCamera->processMouseMovement(xoffset, yoffset);
}

void Application::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app && app->mainCamera) {
        app->mainCamera->processMouseScroll(static_cast<float>(yoffset));
    }
}

void Application::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app) return;

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    if (key == GLFW_KEY_C && action == GLFW_PRESS) {
        app->cameraEnabled = !app->cameraEnabled;
        if (app->cameraEnabled) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            app->firstMouse = true;
            Logger::info("Camera control enabled. Cursor disabled.");
        } else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            Logger::info("Camera control disabled. Cursor normal.");
        }
    }
}

Application::Application()
        : mainWindow(nullptr), mainCamera(nullptr), fluidSimulator(nullptr),
            basicShader(nullptr), particleShader(nullptr), containerMesh(nullptr),
                lastFrameTime(0.0f), lastMouseX(0.0f), lastMouseY(0.0f), firstMouse(true), cameraEnabled(false), particleVAO(0), particleVBO(0) 
    {
}

Application::~Application() {
    Logger::info("Application destructor called. Cleaning up OpenGL resources and GLFW.");

    if (particleVAO != 0) glDeleteVertexArrays(1, &particleVAO);
    if (particleVBO != 0) glDeleteBuffers(1, &particleVBO);

    glfwTerminate();
}

bool Application::initialize() {
    Logger::info("Initializing GLFW...");
    if (!glfwInit()) {
        Logger::error("Failed to initialize GLFW.");
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

    mainWindow = std::make_unique<Window>(1280, 720, "SashFlow Fluid Simulation");
    if (!mainWindow->create()) {
        Logger::error("Failed to create main window.");
        return false;
    }

    mainWindow->makeCurrent();
    glfwSetWindowUserPointer(mainWindow->getGlfwWindow(), this);

    Logger::info("Loading GLAD...");
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        Logger::error("Failed to initialize GLAD.");
        return false;
    }

    int fbWidth, fbHeight;
    glfwGetFramebufferSize(mainWindow->getGlfwWindow(), &fbWidth, &fbHeight);
    glViewport(0, 0, fbWidth, fbHeight);

    glfwSetFramebufferSizeCallback(mainWindow->getGlfwWindow(), framebufferSizeCallback);
    glfwSetCursorPosCallback(mainWindow->getGlfwWindow(), mouseCallback);
    glfwSetScrollCallback(mainWindow->getGlfwWindow(), scrollCallback);
    glfwSetKeyCallback(mainWindow->getGlfwWindow(), keyCallback);

    glfwSetInputMode(mainWindow->getGlfwWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    cameraEnabled = true;

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // Función de blending estándar

    Logger::info("Initializing shaders...");
    basicShader = std::make_unique<Shader>();
    if (!basicShader->load("data/shaders/basic.vert", "data/shaders/basic.frag")) {
        Logger::error("Failed to load basic shader.");
        return false;
    }

    particleShader = std::make_unique<Shader>();
    if (!particleShader->load("data/shaders/particle.vert", "data/shaders/particle.frag")) {
        Logger::error("Failed to load particle shader.");
        return false;
    }

    Logger::info("Initializing camera...");
    mainCamera = std::make_unique<Camera>(glm::vec3(0.0f, 0.0f, 3.0f));
    mainCamera->setAspectRatio((float)fbWidth / (float)fbHeight);

    Logger::info("Creating container mesh with Vertex struct");
    std::vector<Vertex> containerVertices = {
        // Position             Normal                  TexCoords (dummy)
        // Vertex 0: back-bottom-left
        {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},

        // Vertex 1: back-bottom-right
        {{ 1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},

        // Vertex 2: back-top-right
        {{ 1.0f,  1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},

        // Vertex 3: back-top-left
        {{-1.0f,  1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

        // Vertex 4: front-bottom-left
        {{-1.0f, -1.0f,  1.0f}, {0.0f, 0.0f,  1.0f}, {0.0f, 0.0f}},

        // Vertex 5: front-bottom-right
        {{ 1.0f, -1.0f,  1.0f}, {0.0f, 0.0f,  1.0f}, {1.0f, 0.0f}},

        // Vertex 6: front-top-right
        {{ 1.0f,  1.0f,  1.0f}, {0.0f, 0.0f,  1.0f}, {1.0f, 1.0f}},

        // Vertex 7: front-top-left
        {{-1.0f,  1.0f,  1.0f}, {0.0f, 0.0f,  1.0f}, {0.0f, 1.0f}}
    };

    std::vector<unsigned int> containerIndices = {
        // Back face
        0, 1, 2,    // first triangle
        0, 2, 3,    // second triangle

        // Front face
        4, 5, 6,    // first triangle
        4, 6, 7,    // second triangle

        // Left face
        3, 0, 4,    // first triangle (3,0,4)
        3, 4, 7,    // second triangle (3,4,7)

        // Right face
        2, 1, 5,    // first triangle (2,1,5)
        2, 5, 6,    // second triangle (2,5,6)

        // Bottom face
        0, 1, 5,    // first triangle (0,1,5)
        0, 5, 4,    // second triangle (0,5,4)

        // Top face
        3, 2, 6,    // first triangle (3,2,6)
        3, 6, 7     // second triangle (3,6,7)
    };

    std::vector<Texture> containerTextures;
    containerMesh = std::make_unique<Mesh>(containerVertices, containerIndices, containerTextures);


    Logger::info("Initializing SPH fluid simulator...");
    glm::vec3 boxMin = glm::vec3(-1.0f, -1.0f, -1.0f);
    glm::vec3 boxMax = glm::vec3(1.0f, 1.0f, 1.0f);
    // fluidSimulator = std::make_unique<SphFluidModel>(20000, boxMin, boxMax);
    fluidSimulator = std::make_unique<SphFluidModel>(1000, boxMin, boxMax);
    if (!fluidSimulator->initialize()) {
        Logger::error("Failed to initialize SPH fluid model.");
        return false;
    }

    Logger::info("Setting up particle VAO/VBO...");
    glGenVertexArrays(1, &particleVAO);
    glGenBuffers(1, &particleVBO);

    glBindVertexArray(particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, fluidSimulator->getNumParticles() * (3 + 3) * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, (3 + 3) * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, (3 + 3) * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    Logger::info("Application initialization complete.");
    return true;
}

void Application::run() {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        Logger::error("OpenGL Error before main loop: " + std::to_string(err));
    }

    while (!mainWindow->shouldClose()) {
        float currentFrameTime = static_cast<float>(glfwGetTime());
        float deltaTime = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;

        processInput(deltaTime);
        fluidSimulator->update(deltaTime);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glm::mat4 projection = mainCamera->getProjectionMatrix();
        glm::mat4 view = mainCamera->getViewMatrix();
        glm::mat4 model = glm::mat4(1.0f);

        basicShader->use();
        basicShader->setMat4("projection", projection);
        basicShader->setMat4("view", view);
        basicShader->setMat4("model", model);
        basicShader->setVec3("objectColor", glm::vec3(0.5f, 0.5f, 0.5f));

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        containerMesh->draw(*basicShader);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        particleShader->use();
        particleShader->setMat4("projection", projection);
        particleShader->setMat4("view", view);
        particleShader->setMat4("model", model);
        // mainCamera->getFOV() / (float)mainWindow->getHeight() * 1000.0f es una heurística para un tam cte en pantalla.
        particleShader->setFloat("pointSize", mainCamera->getFOV() / (float)mainWindow->getHeight() * 1000.0f);

        const std::vector<float>& particleRenderData = fluidSimulator->getParticleRenderData();
        glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, particleRenderData.size() * sizeof(float), particleRenderData.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindVertexArray(particleVAO);
        glDrawArrays(GL_POINTS, 0, fluidSimulator->getNumParticles());
        glBindVertexArray(0);
        mainWindow->swapBuffers();
        glfwPollEvents();
    }
}

void Application::processInput(float deltaTime) {
    if (!cameraEnabled) return;

    if (mainWindow->isKeyPressed(GLFW_KEY_W))
        mainCamera->processKeyboard(FORWARD, deltaTime);

    if (mainWindow->isKeyPressed(GLFW_KEY_S))
        mainCamera->processKeyboard(BACKWARD, deltaTime);

    if (mainWindow->isKeyPressed(GLFW_KEY_A))
        mainCamera->processKeyboard(LEFT, deltaTime);

    if (mainWindow->isKeyPressed(GLFW_KEY_D))
        mainCamera->processKeyboard(RIGHT, deltaTime);

    if (mainWindow->isKeyPressed(GLFW_KEY_SPACE))
        mainCamera->processKeyboard(UP, deltaTime);

    if (mainWindow->isKeyPressed(GLFW_KEY_LEFT_SHIFT))
        mainCamera->processKeyboard(DOWN, deltaTime);
}

void Application::shutdown() {
    Logger::info("Application shutdown sequence initiated.");
}