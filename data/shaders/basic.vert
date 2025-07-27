#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords; // Añadido para la clase Mesh completa

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 objectColor; // Color uniforme desde la CPU

out vec3 FragPos;
out vec3 Normal;
// out vec2 TexCoords; // No es necesario si no se usa en el fragment shader
out vec3 OutColor; // Para pasar el objectColor o un color base al fragment shader

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
    
    // TexCoords = aTexCoords; // No es necesario si no se usa en el fragment shader
    OutColor = objectColor; // Pasa el color uniforme al fragment shader
}