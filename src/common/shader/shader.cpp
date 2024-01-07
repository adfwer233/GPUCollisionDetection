#include "common/shader/shader.hpp"

#include <sstream>

Shader::Shader(const std::string& vertex_path, const std::string& fragment_path) {
    std::string vertex_shader_source, fragment_shader_source;
    std::ifstream vertex_ifstream, fragment_ifstream;

    auto read_source = [](std::string& target, const std::string& path) {
        std::ifstream input_stream;
        std::stringstream input_string_stream;
        input_stream.open(path);
        input_string_stream << input_stream.rdbuf();
        target = input_string_stream.str();
    };

    read_source(vertex_shader_source, vertex_path);
    read_source(fragment_shader_source, fragment_path);

    unsigned int vertex, fragment;

    int success;
    constexpr int log_len = 512;
    char log[log_len];

    auto vertex_code = vertex_shader_source.c_str();
    auto fragment_code = fragment_shader_source.c_str();

    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vertex_code, nullptr);
    glCompileShader(vertex);

    glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertex, log_len, nullptr, log);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << log << std::endl;
    }

    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fragment_code, nullptr);
    glCompileShader(fragment);

    glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragment, log_len, nullptr, log);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << log << std::endl;
    }

    this->ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);

    glLinkProgram(ID);

    glGetShaderiv(ID, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(ID, log_len, nullptr, log);
        std::cout << "ERROR::SHADER::LINKED::COMPILATION_FAILED\n" << log << std::endl;
    }

    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

void Shader::use() const {
    glUseProgram(this->ID);
}

void Shader::set_float(const std::string &name, float value) const{
    glUniform1fv(glGetUniformLocation(ID, name.c_str()), 1, &value);
}

void Shader::set_int(const std::string &name, int value) const{
    glUniform1iv(glGetUniformLocation(ID, name.c_str()), 1, &value);
}

void Shader::set_vec3(const std::string &name, const glm::vec3 &value) const {
    glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}

void Shader::set_vec4(const std::string &name, const glm::vec4 &value) const{
    glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}

void Shader::set_mat4(const std::string &name, const glm::mat4 &value) const {
    glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &value[0][0]);
}