#include "common/solid/solid.hpp"

Ground::Ground(glm::vec3 t_center) {
    this->center = t_center;
}

Mesh Ground::construct_mesh() {
    Mesh mesh;

    auto A = center + glm::vec3 {-ground_size, 0, -ground_size};
    auto B = center + glm::vec3 {ground_size, 0, -ground_size};
    auto C = center + glm::vec3 {ground_size, 0, ground_size};
    auto D = center + glm::vec3 {-ground_size, 0, ground_size};

    mesh.vertices.emplace_back(A, glm::vec3 {0, 0, 1}, glm::vec2 {0, 0});
    mesh.vertices.emplace_back(B, glm::vec3 {0, 0, 1}, glm::vec2 {1, 0});
    mesh.vertices.emplace_back(C, glm::vec3 {0, 0, 1}, glm::vec2 {1, 1});
    mesh.vertices.emplace_back(D, glm::vec3 {0, 0, 1}, glm::vec2 {0, 1});

    mesh.face_indices.emplace_back(0, 1 ,2);
    mesh.face_indices.emplace_back(0, 2, 3);

    mesh.transform = glm::mat4(1.0f);

    return mesh;
}

bool Ground::is_collision_with(Solid &solid) {
    if (typeid(solid) == typeid(Ball)) {
        Ball &ball = dynamic_cast<Ball &>(solid);
        bool res = (ball.center.y - ball.radius - this->center.y) * (ball.center.y + ball.radius - this->center.y) < 0;

        if (res) {
            ball.center.y = this->center.y + ball.radius + 1e-8;
        }

        return res;
    } else {
        assert(false);
    }
}

AxisAlignedBoundingBox Ground::getBoundingBox() {
    return {center.x - ground_size, center.x + ground_size, center.y, center.y, center.z - ground_size, center.z + ground_size};
}