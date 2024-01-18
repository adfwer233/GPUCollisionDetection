#include <format>
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

    mesh.vertices.emplace_back(A, glm::vec3{0, 1, 0}, glm::vec2{0, 0});
    mesh.vertices.emplace_back(B, glm::vec3{0, 1, 0}, glm::vec2{1, 0});
    mesh.vertices.emplace_back(C, glm::vec3{0, 1, 0}, glm::vec2{1, 1});
    mesh.vertices.emplace_back(D, glm::vec3{0, 1, 0}, glm::vec2{0, 1});

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
            solid.center.x = this->center.x - this->box_size + ball.radius + 1e-8;
            solid.velocity.x *= -1.0f;
        }

        // x-axis, upper side
        if (ball.center.x + ball.radius > this->center.x + this->box_size) {
            res = true;
            solid.center.x = this->center.x + this->box_size - ball.radius - 1e-8;
            solid.velocity.x *= -1.0f;
        }

        // y-axis, lower side
        if (ball.center.y - ball.radius < this->center.y - this->box_size + 1e-6) {
            res = true;
            solid.center.y = this->center.y - this->box_size + ball.radius + 1e-6;
            solid.velocity.y *= -0.5f;
            // std::cout << std::format("bottom {} {} {} {}", solid.velocity.x, solid.velocity.y, solid.velocity.z, ball.center.y) << std::endl;
        }

        // y-axis, upper side
        if (ball.center.y + ball.radius > this->center.y + this->box_size) {
            res = true;
            solid.center.y = this->center.y + this->box_size - ball.radius - 1e-8;
            solid.velocity.y *= -1.0f;
        }

        // z-axis, lower side
        if (ball.center.z - ball.radius < this->center.z - this->box_size) {
            res = true;
            ball.center.z = this->center.z - this->box_size + ball.radius + 1e-8;
            solid.velocity.z *= -1.0f;
        }

        // z-axis, upper side
        if (ball.center.z + ball.radius > this->center.z + this->box_size) {
            res = true;
            ball.center.z = this->center.z + this->box_size - ball.radius - 1e-8;
            solid.velocity.z *= -1.0f;
        }

        return res;
    }
}

AxisAlignedBoundingBox EnvironmentBox::getBoundingBox() {
    return AxisAlignedBoundingBox();
}
