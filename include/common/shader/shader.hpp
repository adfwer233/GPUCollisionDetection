#pragma once

#include "fstream"
#include "glad/glad.h"
#include "glm/glm.hpp"
#include "iostream"
#include "string"

class Shader {
  public:
    unsigned int ID;

    // construct shader from .vs and .fs files
    Shader(const std::string &vertexPath, const std::string &fragmentPath);

    void use() const;

    void set_float(const std::string &name, float value) const;

    void set_int(const std::string &name, int value) const;

    void set_vec3(const std::string &name, const glm::vec3 &value) const;

    void set_vec4(const std::string &name, const glm::vec4 &value) const;

    void set_mat4(const std::string &name, const glm::mat4 &value) const;
};