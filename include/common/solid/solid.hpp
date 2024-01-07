#pragma once

#include "optional"
#include "glm/glm.hpp"

#include "common/simulation/simulation.hpp"
#include "common/mesh/mesh.hpp"

class Solid {
protected:
    double mass;
    glm::vec3 center;
    glm::vec3 velocity;
    std::optional<AxisAlignedBoundingBox> box;
public:

    virtual Mesh construct_mesh() = 0;
    virtual AxisAlignedBoundingBox getBoundingBox() = 0;
    virtual bool is_collision_with(Solid &solid) = 0;
    friend class Simulator;
};

class Ball: public Solid {
private:
    float radius{};
    bool ball_collision_with(Ball &solid);
public:
    Ball() = delete;
    Ball(glm::vec3 t_center, float t_mass, glm::vec3 t_velocity, float t_radius);

    Mesh construct_mesh() override;
    AxisAlignedBoundingBox getBoundingBox() override;
    bool is_collision_with(Solid &solid) override;
};