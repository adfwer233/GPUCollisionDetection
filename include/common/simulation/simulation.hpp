#pragma once

#include "glm/glm.hpp"

#include "common/solid/solid.hpp"
#include "field.hpp"

class Simulation {
private:
    float time_per_frame;
public:
    explicit Simulation(float t_time_per_frame) : time_per_frame(t_time_per_frame) {};
    void update_state_with_field(Solid &solid, const VectorField& field) const;
    void update_state_when_collide(Solid &solid1, Solid &solid2) const;
};