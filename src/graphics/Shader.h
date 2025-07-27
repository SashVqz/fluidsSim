#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <glad/glad.h>
#include <glm/glm.hpp> // Para matrices y vectores GLM
#include <glm/gtc/type_ptr.hpp> // Para glm::value_ptr

class Shader {
public:
    unsigned int ID; // El ID del programa de shader OpenGL

    // Constructor
    // TODO: Carga y compila los shaders desde los archivos de ruta
    Shader();

    // Método para cargar y compilar los shaders
    // Retorna true si la compilación fue exitosa, false en caso contrario
    bool load(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr);

    // Usar el programa de shader
    void use();

    // Métodos uniformes (setters para pasar datos a los shaders)
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setVec2(const std::string& name, const glm::vec2& value) const;
    void setVec2(const std::string& name, float x, float y) const;
    void setVec3(const std::string& name, const glm::vec3& value) const;
    void setVec3(const std::string& name, float x, float y, float z) const;
    void setVec4(const std::string& name, const glm::vec4& value) const;
    void setVec4(const std::string& name, float x, float y, float z, float w) const;
    void setMat2(const std::string& name, const glm::mat2& mat) const;
    void setMat3(const std::string& name, const glm::mat3& mat) const;
    void setMat4(const std::string& name, const glm::mat4& mat) const;

private:
    // Método privado para comprobar errores de compilación/enlazado
    void checkCompileErrors(unsigned int shader, std::string type);
};