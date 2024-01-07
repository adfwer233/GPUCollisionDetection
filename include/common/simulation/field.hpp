#pragma once

#include "glm/glm.hpp"

class VectorField {
public:
    [[nodiscard]] virtual glm::vec3 eval(glm::vec3 position) const = 0;
};

class GravityField : VectorField {
public:
    [[nodiscard]] glm::vec3 eval(glm::vec3 position) const override;
};

class XSymmetricField: VectorField {
public:
    [[nodiscard]] glm::vec3 eval(glm::vec3 position) const override;
};