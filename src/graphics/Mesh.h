#pragma once

#include <glad/glad.h> // Para GLuint, etc.
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <vector>

// Estructura para representar un vértice
struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
    // Opcional: Tangent y Bitangent si los modelos los tienen para normal mapping avanzado
    // glm::vec3 Tangent;
    // glm::vec3 Bitangent;
};

// Estructura para representar una textura
struct Texture {
    unsigned int id;
    std::string type; // e.g., "texture_diffuse", "texture_specular", "texture_normal", "texture_height"
    std::string path; // Ruta de la textura para comparaciones/depuración
};

class Mesh {
public:
    // Datos de la malla
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture> textures;
    unsigned int VAO; // Vertex Array Object

    // Constructor
    Mesh(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices, const std::vector<Texture>& textures);

    // Destructor (¡Importante para liberar recursos de OpenGL!)
    ~Mesh();

    // Dibujar la malla
    void draw(const class Shader& shader) const; // Toma una referencia al shader para establecer los samplers

private:
    // Render data
    unsigned int VBO, EBO; // Vertex Buffer Object, Element Buffer Object

    // Configura los buffers del búfer del vértice/índice
    void setupMesh();
};