#include "common/simulation/field.hpp"

glm::vec3 GravityField::eval(glm::vec3 position) const {
    return {0, -9.8, 0};
}
