#pragma once

#include "common/mesh/mesh.hpp"
#include "common/simulation/simulation.hpp"
#include "glm/glm.hpp"
#include "optional"

/**
 * @brief Base class representing a solid
 */
class Solid {
  public:
    /**
     * @brief mass of solid
     */
    double mass;

    /**
     * @brief mass center of solid, update when simulating
     */
    glm::vec3 center;

    /**
     * @brief velocity of solid, update when simulation
     */
    glm::vec3 velocity;

    /**
     * @brief Axis aligned bounding box of solid
     */
    std::optional<AxisAlignedBoundingBox> box;

    /**
     * @brief optional reference to a mesh for visualization.
     */
    std::optional<std::reference_wrapper<Mesh>> mesh_ref{std::nullopt};

    /**
     * @brief construct mesh of current solid
     * @return mesh of this solid
     */
    virtual Mesh construct_mesh() = 0;

    /**
     * @brief Get
     * @return
     */
    virtual AxisAlignedBoundingBox getBoundingBox() = 0;

    /**
     * Judge if current solid is collided with another solid
     * @param solid
     * @return whether collided
     */
    virtual bool is_collision_with(Solid &solid) = 0;

    /**
     * Simulator can change infos.
     */
    friend class Simulator;
};

/**
 * @brief class represents solid ball
 */
class Ball : public Solid {
  private:
    /**
     * @brief Judge if this ball is collided with another ball
     * @param solid ,actually a ball
     * @return whether collided
     */
    bool ball_collision_with(Ball &solid);

  public:
    /**
     * @brief radius of this ball
     */
    float radius{};

    /**
     * @brief default constructor is deleted
     */
    Ball() = delete;

    /**
     * @brief construct a ball
     * @param t_center
     * @param t_mass
     * @param t_velocity
     * @param t_radius
     */
    Ball(glm::vec3 t_center, float t_mass, glm::vec3 t_velocity, float t_radius);

    Mesh construct_mesh() override;
    AxisAlignedBoundingBox getBoundingBox() override;
    bool is_collision_with(Solid &solid) override;
};

/**
 * @brief Class represents solid ground, i.e. a plane.
 */
class Ground : public Solid {
  private:
    /*
     * @brief size of this ground
     */
    static constexpr float ground_size = 10.0f;

  public:
    /**
     * @brief center of this ground
     * the normal of the ground is {0, 1, 0}
     * @param center
     */
    Ground(glm::vec3 center);

    Mesh construct_mesh() override;
    AxisAlignedBoundingBox getBoundingBox() override;
    bool is_collision_with(Solid &solid) override;
};

/**
 * @brief Class represents the cubic bounding box of our environment
 *
 * Only the bottom side is visualized
 */
class EnvironmentBox : public Solid {
  private:
    /**
     * @brief the size of environment box
     */
    static constexpr float box_size = 10.0f;

  public:
    explicit EnvironmentBox(glm::vec3 center);

    /**
     * @brief get mesh, only have bottom side face
     * @return
     */
    Mesh construct_mesh() override;
    AxisAlignedBoundingBox getBoundingBox() override;
    bool is_collision_with(Solid &solid) override;
};