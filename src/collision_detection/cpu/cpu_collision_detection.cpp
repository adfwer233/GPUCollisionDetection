#include "collision_detection/cpu/cpu_collision_detection.hpp"

#include "glm/glm.hpp"

/**
 * @param solid_ref_vector vector of solid references to perform collision
 * detection.
 */
void CPUNaiveCollisionDetection::collision_detection(
    std::vector<std::reference_wrapper<Solid>> &solid_ref_vector) {
  size_t n = solid_ref_vector.size();

  for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
      auto solid_i = solid_ref_vector[i];
      auto solid_j = solid_ref_vector[j];
      if (solid_i.get().is_collision_with(solid_j)) {
        std::swap(solid_i.get().velocity, solid_j.get().velocity);

        if (typeid(solid_i.get()) == typeid(Ball) and
            typeid(solid_j.get()) == typeid(Ball)) {
          Ball &ball_i = dynamic_cast<Ball &>(solid_i.get());
          Ball &ball_j = dynamic_cast<Ball &>(solid_j.get());
          float center_distance = glm::length(ball_j.center - ball_i.center);
          //                    std::cout << center_distance << std::endl;
          float delta = ball_i.radius + ball_j.radius - center_distance + 1e-6;

          //                    ball_j.center += delta *
          //                    glm::normalize(ball_j.center - ball_i.center);
        }
      }
    }
  }
}