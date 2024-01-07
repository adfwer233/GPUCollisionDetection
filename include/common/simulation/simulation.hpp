#pragma once

#include "glm/glm.hpp"

#include "field.hpp"

class Solid;

class Simulator {
private:
    float time_per_frame;
public:
    explicit Simulator(float t_time_per_frame) : time_per_frame(t_time_per_frame) {};
    glm::vec3 update_state_with_field(Solid &solid, const VectorField& field) const;
    void update_state_when_collide(Solid &solid1, Solid &solid2) const;
};