#include "Mesh.h"
#include "Shader.h" // Necesario para usar la clase Shader en draw()
#include "utils/Logger.h" // Para mensajes de log

// Constructor
Mesh::Mesh(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices, const std::vector<Texture>& textures) {
    this->vertices = vertices;
    this->indices = indices;
    this->textures = textures;

    // Ahora configurar los buffers del objeto
    setupMesh();
    Logger::info("Mesh created with " + std::to_string(vertices.size()) + " vertices and " + std::to_string(indices.size()) + " indices.");
}

// Destructor
Mesh::~Mesh() {
    // Liberar los recursos de OpenGL
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO); // También borrar el EBO
    Logger::info("Mesh destroyed. VAO, VBO, EBO deleted.");
}

// Dibujar la malla
void Mesh::draw(const Shader& shader) const {
    // Enlazar texturas apropiadas
    unsigned int diffuseNr = 1;
    unsigned int specularNr = 1;
    unsigned int normalNr = 1;
    unsigned int heightNr = 1;

    for (unsigned int i = 0; i < textures.size(); i++) {
        glActiveTexture(GL_TEXTURE0 + i); // Activar la unidad de textura apropiada antes de enlazar
        // Obtener el número de textura (ej. "texture_diffuse1")
        std::string number;
        std::string name = textures[i].type;
        if (name == "texture_diffuse")
            number = std::to_string(diffuseNr++);
        else if (name == "texture_specular")
            number = std::to_string(specularNr++); // Incrementa specularNr
        else if (name == "texture_normal")
            number = std::to_string(normalNr++); // Incrementa normalNr
        else if (name == "texture_height")
            number = std::to_string(heightNr++); // Incrementa heightNr
        else {
            Logger::warning("Unknown texture type: " + name);
        }

        // Ahora establece el sampler a la unidad de textura correcta en el shader
        // Asegúrate de que tu shader tiene uniforms como 'texture_diffuse1', 'texture_specular1', etc.
        shader.setInt((name + number).c_str(), i);
        // Y enlaza la textura
        glBindTexture(GL_TEXTURE_2D, textures[i].id);
    }

    // Dibujar malla
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // Siempre es una buena práctica activar la textura predeterminada (o ninguna) de nuevo después de usarla
    // para evitar efectos secundarios en otros objetos.
    glActiveTexture(GL_TEXTURE0);
}

// Configurar los buffers del búfer del vértice/índice
void Mesh::setupMesh() {
    // Crear VAO, VBO, EBO
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    // Cargar datos en el VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

    // Cargar datos en el EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    // Configurar los punteros de atributo del vértice
    // Posiciones (layout (location = 0))
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    // Normales (layout (location = 1))
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
    // Coordenadas de textura (layout (location = 2))
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
    // Si tuvieras Tangent y Bitangent, irían aquí en locations 3 y 4

    glBindVertexArray(0); // Desenlazar VAO
}