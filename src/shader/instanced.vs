#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTextureCoord;
layout (location = 3) in mat4 aInstanceMatrix;

out vec3 Normal;
out vec3 FragPos;
out vec2 TextureCoord;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(aInstanceMatrix * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(aInstanceMatrix))) * aNormal;

    gl_Position = projection * view * vec4(FragPos, 1.0);
}