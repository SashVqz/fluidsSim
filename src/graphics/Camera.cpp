#include "Camera.h"
#include <iostream> // Solo para depuración, puedes eliminarlo

// Constructor con vectores
Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
    : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM), aspectRatio(16.0f / 9.0f) { // Inicializa aspectRatio
    Position = position;
    WorldUp = up;
    Yaw = yaw;
    Pitch = pitch;
    updateCameraVectors();
}

// Constructor con valores escalares
Camera::Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch)
    : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM), aspectRatio(16.0f / 9.0f) { // Inicializa aspectRatio
    Position = glm::vec3(posX, posY, posZ);
    WorldUp = glm::vec3(upX, upY, upZ);
    Yaw = yaw;
    Pitch = pitch;
    updateCameraVectors();
}

// Retorna la matriz de vista
glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(Position, Position + Front, Up);
}

// Calcula la matriz de proyección (NO NECESITA screenWidth/Height AHORA)
glm::mat4 Camera::getProjectionMatrix() const {
    // Usa el aspectRatio guardado en el miembro de la clase
    return glm::perspective(glm::radians(Zoom), aspectRatio, 0.1f, 100.0f);
}

// Nueva función para establecer la relación de aspecto
void Camera::setAspectRatio(float newAspectRatio) {
    aspectRatio = newAspectRatio;
}

// Procesa el input del teclado
void Camera::processKeyboard(Camera_Movement direction, float deltaTime) {
    float velocity = MovementSpeed * deltaTime;
    if (direction == FORWARD)
        Position += Front * velocity;
    if (direction == BACKWARD)
        Position -= Front * velocity;
    if (direction == LEFT)
        Position -= Right * velocity;
    if (direction == RIGHT)
        Position += Right * velocity;
    if (direction == UP)
        Position += WorldUp * velocity; // Mover directamente hacia arriba en el eje Y global
    if (direction == DOWN)
        Position -= WorldUp * velocity; // Mover directamente hacia abajo en el eje Y global
}

// Procesa el movimiento del ratón
void Camera::processMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch) {
    xoffset *= MouseSensitivity;
    yoffset *= MouseSensitivity;

    Yaw += xoffset;
    Pitch += yoffset;

    // Asegurarse de que cuando el pitch está fuera de los límites, la pantalla no se voltee
    if (constrainPitch) {
        if (Pitch > 89.0f)
            Pitch = 89.0f;
        if (Pitch < -89.0f)
            Pitch = -89.0f;
    }

    // Actualizar los vectores Front, Right y Up usando los ángulos de Euler actualizados
    updateCameraVectors();
}

// Procesa el scroll del ratón
void Camera::processMouseScroll(float yoffset) {
    Zoom -= (float)yoffset;
    if (Zoom < 1.0f)
        Zoom = 1.0f;
    if (Zoom > 45.0f) // El FOV máximo suele ser 45.0f o 90.0f. Aquí lo limitamos a 45.0f.
        Zoom = 45.0f;
}

// Calcula los vectores Front, Right y Up a partir de los ángulos de Euler (roll, pitch, yaw) actualizados de la cámara.
void Camera::updateCameraVectors() {
    // Calcular el nuevo vector Front
    glm::vec3 front;
    front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    front.y = sin(glm::radians(Pitch));
    front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    Front = glm::normalize(front);
    // Recalcular los vectores Right y Up
    Right = glm::normalize(glm::cross(Front, WorldUp));
    Up = glm::normalize(glm::cross(Right, Front));
}