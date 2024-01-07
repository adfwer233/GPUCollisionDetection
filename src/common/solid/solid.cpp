#pragma once

#include "typeinfo"

#include "solid/solid.hpp"

bool Ball::ball_collision_with(Ball &solid) {
    auto distance  = glm::length(this->center - solid.center);
    return distance < this->radius + solid.radius;
}

AxisAlignedBoundingBox Ball::getBoundingBox() {
    return {center.x - radius, center.x + radius, center.y - radius, center.y + radius, center.z - radius, center.z + radius};
}

bool Ball::is_collision_with(Solid &solid) {
    if (typeid(solid) == typeid(Ball)) {
        return this->ball_collision_with(reinterpret_cast<Ball &>(solid));
    } else {
        assert(false);
    }
}

