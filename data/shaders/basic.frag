#version 460 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
// in vec2 TexCoords; // Ya no se recibe si no se envía desde el vertex shader
in vec3 OutColor; // Recibe el color desde el vertex shader

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
// uniform sampler2D texture_diffuse1; // Descomentar si usas texturas

void main()
{
    // Ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    // Combina todos los componentes con el color del objeto
    vec3 finalColor = (ambient + diffuse + specular) * OutColor; // Usa el OutColor del vertex shader
    // Si tuvieras textura_diffuse1:
    // vec3 finalColor = (ambient + diffuse + specular) * vec3(texture(texture_diffuse1, TexCoords));

    FragColor = vec4(finalColor, 1.0);
}