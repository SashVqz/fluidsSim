#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

// Enumeración para el movimiento de la cámara
enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

// Valores predeterminados de la cámara
const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 2.5f;
const float SENSITIVITY = 0.1f;
const float ZOOM = 45.0f; // Este es el FOV predeterminado

class Camera {
public:
    // Atributos de la cámara
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    // Ángulos de Euler
    float Yaw;
    float Pitch;
    // Opciones de la cámara
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom; // FOV (Field of View)

    // Constructor con vectores
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH);
    // Constructor con valores escalares
    Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch);

    // Retorna la matriz de vista calculada usando los ángulos de Euler de la cámara y el lookAt Matrix
    glm::mat4 getViewMatrix() const;

    // Calcula la matriz de proyección (perspectiva)
    // NOTA: En la implementación de Application.cpp, la cámara se inicializa sin screenWidth/Height
    // y luego se usa setAspectRatio en framebufferSizeCallback.
    // Necesitamos un getter para el FOV (Zoom)
    glm::mat4 getProjectionMatrix() const; // Modificado: sin screenWidth/Height aquí
    void setAspectRatio(float aspectRatio); // ¡Nuevo! Para ajustar la relación de aspecto dinámicamente

    // Retorna el valor actual del FOV (Zoom) de la cámara.
    float getFOV() const { return Zoom; } // ¡Nuevo! Getter para el FOV/Zoom

    // Procesa la entrada recibida de cualquier sistema de entrada de teclado. Acepta un ENUM de movimiento de cámara definido.
    void processKeyboard(Camera_Movement direction, float deltaTime);

    // Procesa la entrada recibida de un evento de movimiento del ratón. Espera los valores de offset x e y del último y actual movimiento del ratón.
    void processMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true);

    // Procesa la entrada recibida de un evento de scroll del ratón. Sólo espera la entrada de la rueda de scroll vertical.
    void processMouseScroll(float yoffset);

private:
    // Aspect ratio para la matriz de proyección.
    float aspectRatio; // ¡Nuevo!

    // Calcula los vectores Front, Right y Up a partir de los ángulos de Euler (roll, pitch, yaw) actualizados de la cámara.
    void updateCameraVectors();
};