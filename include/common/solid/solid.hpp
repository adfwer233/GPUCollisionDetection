#pragma once

#include "common/mesh/mesh.hpp"
#include "common/simulation/simulation.hpp"
#include "glm/glm.hpp"
#include "optional"

class Solid {
 public:
  double mass;
  glm::vec3 center;
  glm::vec3 velocity;
  std::optional<AxisAlignedBoundingBox> box;

  std::optional<std::reference_wrapper<Mesh>> mesh_ref{std::nullopt};

  virtual Mesh construct_mesh() = 0;
  virtual AxisAlignedBoundingBox getBoundingBox() = 0;
  virtual bool is_collision_with(Solid &solid) = 0;
  friend class Simulator;
};

class Ball : public Solid {
 private:
  bool ball_collision_with(Ball &solid);

 public:
  float radius{};
  Ball() = delete;
  Ball(glm::vec3 t_center, float t_mass, glm::vec3 t_velocity, float t_radius);

  Mesh construct_mesh() override;
  AxisAlignedBoundingBox getBoundingBox() override;
  bool is_collision_with(Solid &solid) override;
};

class Ground : public Solid {
 private:
  static constexpr float ground_size = 10.0f;

 public:
  Ground(glm::vec3 center);
  Mesh construct_mesh() override;
  AxisAlignedBoundingBox getBoundingBox() override;
  bool is_collision_with(Solid &solid) override;
};