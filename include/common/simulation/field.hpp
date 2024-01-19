#pragma once

#include "glm/glm.hpp"

/**
 * @brief base class of vector fields
 */
class VectorField {
  public:
    [[nodiscard]] virtual glm::vec3 eval(glm::vec3 position) const = 0;
};

/**
 * @brief Gravity Field along y-axis, constant value {0, -1, 0}
 */
class GravityField : public VectorField {
  public:
    [[nodiscard]] glm::vec3 eval(glm::vec3 position) const override;
};

/**
 * @brief Symmetric field for x-axis
 */
class XSymmetricField : public VectorField {
  public:
    [[nodiscard]] glm::vec3 eval(glm::vec3 position) const override;
};