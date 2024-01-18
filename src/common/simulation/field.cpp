#include <iostream>
#include "common/simulation/field.hpp"

glm::vec3 GravityField::eval(glm::vec3 position) const {
    return {0, -1, 0};
}

glm::vec3 XSymmetricField::eval(glm::vec3 position) const {
    if (position.x > 0) {
        return {-1, 0, 0};
    } else {
        return {1, 0, 0};
    }
}