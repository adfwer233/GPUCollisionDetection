#include "common/solid/solid.hpp"

EnvironmentBox::EnvironmentBox(glm::vec3 center) {
    this->center = center;
}

Mesh EnvironmentBox::construct_mesh() {
    Mesh mesh;

    auto A = center + glm::vec3{-box_size, -box_size, -box_size};
    auto B = center + glm::vec3{box_size, -box_size, -box_size};
    auto C = center + glm::vec3{box_size, -box_size, box_size};
    auto D = center + glm::vec3{-box_size, -box_size, box_size};

    mesh.vertices.emplace_back(A, glm::vec3{0, 0, 1}, glm::vec2{0, 0});
    mesh.vertices.emplace_back(B, glm::vec3{0, 0, 1}, glm::vec2{1, 0});
    mesh.vertices.emplace_back(C, glm::vec3{0, 0, 1}, glm::vec2{1, 1});
    mesh.vertices.emplace_back(D, glm::vec3{0, 0, 1}, glm::vec2{0, 1});

    mesh.face_indices.emplace_back(0, 1, 2);
    mesh.face_indices.emplace_back(0, 2, 3);

    mesh.transform = glm::mat4(1.0f);

    return mesh;
}

bool EnvironmentBox::is_collision_with(Solid &solid) {
    if (typeid(solid) == typeid(Ball)) {
        Ball &ball = dynamic_cast<Ball &>(solid);

        bool res = false;

        // x-axis, lower side
        if (ball.center.x - ball.radius < this->center.x - this->box_size) {
            res = true;
            ball.center.x = this->center.x - this->box_size + ball.radius + 1e-8;
        }

        // x-axis, upper side
        if (ball.center.x + ball.radius > this->center.x + this->box_size) {
            res = true;
            ball.center.x = this->center.x + this->box_size - ball.radius - 1e-8;
        }

        // y-axis, lower side
        if (ball.center.y - ball.radius < this->center.y - this->box_size) {
            res = true;
            ball.center.y = this->center.y - this->box_size + ball.radius + 1e-8;
        }

        // y-axis, upper side
        if (ball.center.y + ball.radius > this->center.y + this->box_size) {
            res = true;
            ball.center.y = this->center.y + this->box_size - ball.radius - 1e-8;
        }

        // z-axis, lower side
        if (ball.center.z - ball.radius < this->center.z - this->box_size) {
            res = true;
            ball.center.z = this->center.z - this->box_size + ball.radius + 1e-8;
        }

        // z-axis, upper side
        if (ball.center.z + ball.radius > this->center.z + this->box_size) {
            res = true;
            ball.center.z = this->center.z + this->box_size - ball.radius - 1e-8;
        }

        return res;
    }
}

AxisAlignedBoundingBox EnvironmentBox::getBoundingBox() {
    return AxisAlignedBoundingBox();
}
