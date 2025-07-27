#version 460 core
out vec4 FragColor;

in vec3 ParticleColor;

void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    float distance = length(coord);
    
    if(distance > 0.5)
        discard;
    
    float alpha = 1.0 - smoothstep(0.3, 0.5, distance);
    FragColor = vec4(ParticleColor, alpha);
}