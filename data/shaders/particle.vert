#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float pointSize; // Tama�o del punto en p�xeles

out vec3 particleColor;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    gl_PointSize = pointSize; // Establece el tama�o del punto en p�xeles
    particleColor = aColor;
}