#pragma once

#include "vector"

#include "common/solid/solid.hpp"

class CPUNaiveCollisionDetection {
public:
    static void collision_detection (std::vector<std::reference_wrapper<Solid>> &solid_ref_vector);
};

class CPUSweepAndPruneCollisionDetection {
private:
    struct SolidInfo {
        size_t id;
        AxisAlignedBoundingBox box;
    };
public:
    static void collision_detection (std::vector<std::reference_wrapper<Solid>> &solid_ref_vector, float time_step);
};