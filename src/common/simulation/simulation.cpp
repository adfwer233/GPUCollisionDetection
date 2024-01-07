#include "common/simulation/simulation.hpp"

void Simulation::update_state_when_collide(Solid &solid1, Solid &solid2) const {
    solid1.velocity *= -1.0f;
    solid2.velocity *= -1.0f;

    if (solid1.is_collision_with(solid2)) {
        // ?
    }
}

void Simulation::update_state_with_field(Solid &solid, const VectorField &field) const {
    auto acceleration = field.eval(solid.center);

    auto next_velocity = solid.velocity + acceleration * time_per_frame;
    auto average_velocity = (solid.velocity + next_velocity) / 2.0f;

    solid.center = solid.center + average_velocity * time_per_frame;
    solid.velocity = next_velocity;
}