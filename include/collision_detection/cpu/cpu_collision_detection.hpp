#pragma once

#include "vector"

#include "common/solid/solid.hpp"

class CPUNaiveCollisionDetection {
public:

    static void collision_detection (std::vector<std::reference_wrapper<Solid>> &solid_ref_vector);
};