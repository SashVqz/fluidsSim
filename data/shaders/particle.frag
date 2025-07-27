#version 460 core
out vec4 FragColor;

in vec3 particleColor;

void main()
{
    // Opcional: Para hacer los puntos circulares, puedes usar gl_PointCoord
    // float dist = distance(gl_PointCoord, vec2(0.5, 0.5));
    // if (dist > 0.5)
    //     discard; // Descarta fragmentos fuera del círculo

    FragColor = vec4(particleColor, 1.0);
}