#include "algorithm"

#include "collision_detection/cpu/cpu_collision_detection.hpp"

/**
 * @brief CPU Version of sweep and prune algorithm
 *
 * @note if two ball collide, we will compute the time of collision and compute the final position of the ball
 *
 * @param solid_ref_vector vector of solids to perform collision detection
 * @param time_step time step of current simulation procedure
 */
void CPUSweepAndPruneCollisionDetection::collision_detection(std::vector<std::reference_wrapper<Solid>> &solid_ref_vector, float time_step) {
    /*
     * Step1: construct solid info
     */

    std::vector<SolidInfo> solid_infos;
    for (size_t i = 0; i < solid_ref_vector.size(); i++) {
        SolidInfo info;
        info.id = i;
        info.box = solid_ref_vector[i].get().getBoundingBox();
        solid_infos.push_back(info);
    }

    /*
     * Step2: sort on x-axis (increase by min_x)
     */
    std::ranges::sort(solid_infos, [](SolidInfo &s1, SolidInfo &s2) -> bool {
       return s1.box.min_x < s2.box.min_x;
    });

    /*
     * Step3: get possible collision upper bound via binary search
     */
    std::vector<size_t> possible_collision_upper_bound(solid_infos.size());
    for (size_t i = 0; i < solid_ref_vector.size(); i++) {
        auto it = std::upper_bound(solid_infos.begin(), solid_infos.end(), solid_infos[i].box.max_x, [](float x, SolidInfo &y) -> bool {
            return x < y.box.min_x;
        });
        possible_collision_upper_bound[i] = std::distance(solid_infos.begin(), it);
    }

    /*
     * step4: find all AABB collision pairs
     */
    std::vector<std::vector<size_t>> aabb_collision_result(solid_ref_vector.size());
    for (size_t  i = 0; i < solid_ref_vector.size(); i++) {
        auto box_i = solid_infos[i].box;
        for (size_t j = i + 1; j < possible_collision_upper_bound[i]; j++) {
            auto box_j = solid_infos[j].box;
            bool overlap_flag = false;
            if (std::max(box_i.min_x, box_j.min_x) <= std::min(box_i.max_x, box_j.max_x)) overlap_flag = true;
            if (std::max(box_i.min_y, box_j.min_y) <= std::min(box_i.max_y, box_j.max_y)) overlap_flag = true;
            if (std::max(box_i.min_z, box_j.min_z) <= std::min(box_i.max_z, box_j.max_z)) overlap_flag = true;

            if (overlap_flag) aabb_collision_result[solid_infos[i].id].push_back(solid_infos[j].id);
        }
    }

    /*
     * Step5: handle collision between balls, update state
     */

    for (size_t i = 0; i < solid_ref_vector.size(); i++) {
        for (auto j: aabb_collision_result[i]) {
            auto &solid_i = solid_ref_vector[i].get();
            auto &solid_j = solid_ref_vector[j].get();
            if (solid_i.is_collision_with(solid_j)) {
                glm::vec3 previous_i_center = solid_i.center - time_step * solid_i.velocity;
                glm::vec3 previous_j_center = solid_j.center - time_step * solid_j.velocity;

                Ball &ball_i = static_cast<Ball &>(solid_i);
                Ball &ball_j = static_cast<Ball &>(solid_j);

                float previous_distance = glm::length(previous_i_center - previous_j_center) - ball_i.radius - ball_j.radius;
                float relative_speed = glm::length(solid_i.velocity - solid_j.velocity);

                float collision_time = previous_distance / relative_speed;

                if (collision_time < time_step) {
                    auto collision_displacement_i = -solid_i.velocity * (time_step - collision_time) +
                                                    solid_j.velocity * (time_step - collision_time);
                    auto collision_displacement_j = -solid_j.velocity * (time_step - collision_time) +
                                                    solid_i.velocity * (time_step - collision_time);

                    if (solid_i.mesh_ref.has_value() and solid_j.mesh_ref.has_value()) {
                        solid_i.mesh_ref->get().transform = glm::translate(solid_i.mesh_ref->get().transform,
                                                                           collision_displacement_i);
                        solid_j.mesh_ref->get().transform = glm::translate(solid_j.mesh_ref->get().transform,
                                                                           collision_displacement_j);
                    }

                    solid_i.center += collision_displacement_i;
                    solid_j.center += collision_displacement_j;
                }
                std::swap(solid_i.velocity, solid_j.velocity);
            }
        }
    }
}