#pragma once

#include "common/solid/solid.hpp"

#include "numbers"
#include "ranges"
#include "typeinfo"

bool Ball::ball_collision_with(Ball &solid) {
  auto distance = glm::length(this->center - solid.center);
  return distance < this->radius + solid.radius;
}

AxisAlignedBoundingBox Ball::getBoundingBox() {
  return {center.x - radius, center.x + radius, center.y - radius,
          center.y + radius, center.z - radius, center.z + radius};
}

bool Ball::is_collision_with(Solid &solid) {
  if (typeid(solid) == typeid(Ball)) {
    return this->ball_collision_with(reinterpret_cast<Ball &>(solid));
  } else {
    assert(false);
  }
}

Mesh Ball::construct_mesh() {
  using namespace std::numbers;

  Mesh model;

  constexpr int theta_segments = 10;
  constexpr int phi_segments = 10;

  // Generate Vertices
  for (int i : std::views::iota(0, theta_segments + 1)) {
    for (int j : std::views::iota(0, phi_segments + 1)) {
      double theta = pi * (1.0f * i / theta_segments);
      double phi = 2 * pi * (1.0f * j / phi_segments);

      float x = std::sin(theta) * std::cos(phi);
      float y = std::sin(theta) * std::sin(phi);
      float z = std::cos(theta);

      model.vertices.push_back(
          {{x * radius, y * radius, z * radius},
           {x, y, z},
           {1.0f * i / theta_segments, 1.0f * j / phi_segments}});
    }
  }

  // Generate Indices
  for (unsigned int i = 0; i < theta_segments; i++) {
    for (unsigned int j = 0; j < phi_segments; j++) {
      auto ind = [&](auto x, auto y) { return x * (theta_segments + 1) + y; };
      model.face_indices.push_back(
          {ind(i, j), ind(i + 1, j), ind(i + 1, j + 1)});
      model.face_indices.push_back(
          {ind(i, j), ind(i + 1, j + 1), ind(i, j + 1)});
    }
  }

  model.transform = glm::identity<glm::mat4>();

  //    model.transform = glm::scale(model.transform, glm::vec3(radius, radius,
  //    radius));
  model.transform =
      glm::translate(model.transform, glm::vec3(center.x, center.y, center.z));

  model.box = AxisAlignedBoundingBox({-1, 1, -1, 1, -1, 1});
  return model;
}

Ball::Ball(glm::vec3 t_center, float t_mass, glm::vec3 t_velocity,
           float t_radius)
    : radius(t_radius) {
  this->center = t_center;
  this->mass = t_mass;
  this->velocity = t_velocity;
}
