#include "Application.h"
#include "utils/Logger.h"
#include "graphics/Window.h"
#include "graphics/Shader.h"
#include "graphics/Camera.h"
#include "graphics/Mesh.h" // Incluir la clase Mesh completa
#include "physics/SphFluidModel.h" // Incluir el modelo de fluidos SPH

// Incluir GLAD para glViewport y otras funciones de OpenGL
#include <glad/glad.h>
// Incluir GLFW para inicialización y funciones de eventos
#include <gLFW/glfw3.h>
// Incluir GLM para cálculos de matrices
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector> // Necesario para std::vector

// --- Callbacks globales de GLFW ---
// Estos callbacks son estáticos y requieren un puntero a la instancia de Application
// para interactuar con los miembros de la clase.
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
    float yoffset = static_cast<float>(app->lastMouseY - ypos); // Invertido para ejes Y de OpenGL

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
            app->firstMouse = true; // Resetear firstMouse al habilitar
            Logger::info("Camera control enabled. Cursor disabled.");
        }
        else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            Logger::info("Camera control disabled. Cursor normal.");
        }
    }
}


// --- Implementación de la clase Application ---
Application::Application()
    : mainWindow(nullptr), mainCamera(nullptr), fluidSimulator(nullptr),
    basicShader(nullptr), particleShader(nullptr), containerMesh(nullptr),
    lastFrameTime(0.0f), lastMouseX(0.0f), lastMouseY(0.0f), firstMouse(true), cameraEnabled(false),
    particleVAO(0), particleVBO(0) // Inicializar nuevos miembros
{
}

Application::~Application() {
    Logger::info("Application destructor called. Cleaning up OpenGL resources and GLFW.");
    // Liberar recursos de OpenGL
    if (particleVAO != 0) glDeleteVertexArrays(1, &particleVAO);
    if (particleVBO != 0) glDeleteBuffers(1, &particleVBO);

    // Los unique_ptr se encargarán de liberar mainWindow, mainCamera, etc.
    // glfwTerminate() debe llamarse al final, idealmente en el destructor de Application
    // o al final de main(). Lo mantendremos aquí.
    glfwTerminate();
}

bool Application::initialize() {
    // 1. Inicializar GLFW
    Logger::info("Initializing GLFW...");
    if (!glfwInit()) {
        Logger::error("Failed to initialize GLFW.");
        return false;
    }

    // Configuración de GLFW para OpenGL 4.6 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Requerido para macOS
#endif

    // 2. Crear la ventana
    mainWindow = std::make_unique<Window>(1280, 720, "SashFlow Fluid Simulation");
    if (!mainWindow->create()) {
        Logger::error("Failed to create main window.");
        return false;
    }

    mainWindow->makeCurrent();
    // ¡Muy importante! Asociar la instancia de Application con la ventana GLFW
    glfwSetWindowUserPointer(mainWindow->getGlfwWindow(), this);

    // 3. Cargar GLAD
    Logger::info("Loading GLAD...");
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        Logger::error("Failed to initialize GLAD.");
        return false;
    }

    // Configurar viewport inicial
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(mainWindow->getGlfwWindow(), &fbWidth, &fbHeight);
    glViewport(0, 0, fbWidth, fbHeight);

    // Registrar callbacks
    glfwSetFramebufferSizeCallback(mainWindow->getGlfwWindow(), framebufferSizeCallback);
    glfwSetCursorPosCallback(mainWindow->getGlfwWindow(), mouseCallback);
    glfwSetScrollCallback(mainWindow->getGlfwWindow(), scrollCallback);
    glfwSetKeyCallback(mainWindow->getGlfwWindow(), keyCallback);

    // Deshabilitar el cursor inicialmente y habilitar control de cámara
    glfwSetInputMode(mainWindow->getGlfwWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    cameraEnabled = true;

    // Habilitar el test de profundidad
    glEnable(GL_DEPTH_TEST);
    // Habilitar puntos de tamaño programable en el shader (para partículas)
    glEnable(GL_PROGRAM_POINT_SIZE);
    // Habilitar blending para efectos de transparencia (si los shaders lo usan)
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // Función de blending estándar

    // 4. Inicializar shaders
    Logger::info("Initializing shaders...");
    basicShader = std::make_unique<Shader>("data/shaders/basic.vert", "data/shaders/basic.frag");
    if (!basicShader->load()) {
        Logger::error("Failed to load basic shader.");
        return false;
    }

    particleShader = std::make_unique<Shader>("data/shaders/particle.vert", "data/shaders/particle.frag");
    if (!particleShader->load()) {
        Logger::error("Failed to load particle shader.");
        return false;
    }

    // 5. Inicializar cámara
    Logger::info("Initializing camera...");
    mainCamera = std::make_unique<Camera>(glm::vec3(0.0f, 0.0f, 3.0f));
    mainCamera->setAspectRatio((float)fbWidth / (float)fbHeight); // Configurar relación de aspecto inicial

    // 6. Crear malla del contenedor (AHORA USANDO LA CLASE MESH COMPLETA CON VERTEX STRUCT)
    Logger::info("Creating container mesh with Vertex struct...");

    // 8 vértices únicos para un cubo
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

    // 36 índices para dibujar los 12 triángulos (2 por cara) de un cubo
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

    std::vector<Texture> containerTextures; // No necesitamos texturas para el contenedor wireframe

    // Crear la instancia de Mesh con los nuevos datos de vértices y sin texturas
    containerMesh = std::make_unique<Mesh>(containerVertices, containerIndices, containerTextures);


    // 7. Inicializar simulador de fluido SPH
    Logger::info("Initializing SPH fluid simulator...");
    glm::vec3 boxMin = glm::vec3(-1.0f, -1.0f, -1.0f); // Esquina inferior izquierda trasera
    glm::vec3 boxMax = glm::vec3(1.0f, 1.0f, 1.0f);   // Esquina superior derecha delantera
    fluidSimulator = std::make_unique<SphFluidModel>(20000, boxMin, boxMax); // 20.000 partículas
    if (!fluidSimulator->initialize()) {
        Logger::error("Failed to initialize SPH fluid model.");
        return false;
    }

    // 8. Configurar VAO/VBO para partículas
    Logger::info("Setting up particle VAO/VBO...");
    glGenVertexArrays(1, &particleVAO);
    glGenBuffers(1, &particleVBO);

    glBindVertexArray(particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);

    // Asigna espacio suficiente para las posiciones y colores de todas las partículas.
    // Cada partícula: 3 floats para posición, 3 floats para color.
    glBufferData(GL_ARRAY_BUFFER, fluidSimulator->getNumParticles() * (3 + 3) * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    // Atributo de posición (layout (location = 0) en particle.vert)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, (3 + 3) * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Atributo de color (layout (location = 1) en particle.vert)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, (3 + 3) * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0); // Desenlazar VBO
    glBindVertexArray(0);             // Desenlazar VAO

    Logger::info("Application initialization complete.");
    return true;
}

void Application::run() {
    // Para depuración, comprueba los errores de OpenGL al inicio del bucle
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        Logger::error("OpenGL Error before main loop: " + std::to_string(err));
    }

    while (!mainWindow->shouldClose()) {
        // Calcular delta time
        float currentFrameTime = static_cast<float>(glfwGetTime());
        float deltaTime = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;

        // Procesar entrada del usuario
        processInput(deltaTime);

        // --- Actualizar simulación ---
        fluidSimulator->update(deltaTime);

        // --- Renderizado ---
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f); // Color de fondo gris azulado
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 projection = mainCamera->getProjectionMatrix();
        glm::mat4 view = mainCamera->getViewMatrix();
        glm::mat4 model = glm::mat4(1.0f); // Matriz de modelo identidad para objetos en el origen

        // 1. Renderizar el contenedor (como un cubo wireframe)
        basicShader->use();
        basicShader->setMat4("projection", projection);
        basicShader->setMat4("view", view);
        basicShader->setMat4("model", model);
        basicShader->setVec3("objectColor", glm::vec3(0.5f, 0.5f, 0.5f)); // Gris para el contenedor

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); // Dibujar solo el contorno (wireframe)
        containerMesh->draw(*basicShader); // Pasar el shader al método draw de Mesh
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // Volver al modo de relleno para futuros objetos

        // 2. Renderizar las partículas SPH
        particleShader->use();
        particleShader->setMat4("projection", projection);
        particleShader->setMat4("view", view);
        particleShader->setMat4("model", model);
        // Ajusta el tamaño de la partícula. Puedes ajustar 1000.0f a tu gusto.
        // mainCamera->getFOV() / (float)mainWindow->getHeight() * 1000.0f es una heurística para un tamaño constante en pantalla.
        particleShader->setFloat("pointSize", mainCamera->getFOV() / (float)mainWindow->getHeight() * 1000.0f);

        // Obtener los datos de las partículas del simulador
        const std::vector<float>& particleRenderData = fluidSimulator->getParticleRenderData();

        // Actualizar el contenido del VBO con los datos de las partículas
        glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, particleRenderData.size() * sizeof(float), particleRenderData.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Dibujar las partículas como puntos
        glBindVertexArray(particleVAO);
        glDrawArrays(GL_POINTS, 0, fluidSimulator->getNumParticles());
        glBindVertexArray(0);

        // Intercambiar buffers y sondear eventos
        mainWindow->swapBuffers();
        glfwPollEvents();
    }
}

void Application::processInput(float deltaTime) {
    // Si la cámara está deshabilitada, no procesamos la entrada de movimiento de la cámara
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
    // Los destructores de los unique_ptr y el destructor de Application
    // se encargarán de liberar la mayoría de los recursos.
    // glfwTerminate() ya se llama en el destructor de Application.
}