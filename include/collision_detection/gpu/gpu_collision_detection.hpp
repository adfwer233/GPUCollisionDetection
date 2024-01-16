#pragma once

#include "common/solid/solid.hpp"
#include "vector"

struct SolidInfoGPU {
  size_t id;
  float min_x, max_x, min_y, max_y, min_z, max_z;
};

/*
 * @brief Class implementing GPU SaP Algorithm
 */
class GPUSweepAndPruneCollisionDetection {
 public:
  static void collision_detection(
      std::vector<std::reference_wrapper<Solid>> &solid_ref_vector,
      float time_step);
};