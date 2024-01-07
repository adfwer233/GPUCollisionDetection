#pragma once

#include "vector"
#include "optional"

#include "common/camera/camera.hpp"
#include "common/shader/shader.hpp"

#include "glm/glm.hpp"

struct AxisAlignedBoundingBox {
    float min_x, max_x, min_y, max_y, min_z, max_z;
};

struct TriangleVerticeIndex {
    unsigned int x, y, z;
};

struct TriangleWithNormal {
    glm::vec3 point, normal;
    glm::vec2 texture_coord;
};

class Mesh {

public:
    std::vector<TriangleWithNormal> vertices;
    std::vector<TriangleVerticeIndex> face_indices;
    glm::mat4 transform;

    glm::vec3 object_color {1, 1, 1};

    unsigned int VAO {}, VBO {}, EBO {};

    bool strip_flag {false};

    void bind_buffer();

    void process_rendering(Shader& shader, Camera camera, glm::vec3 lightPos);

    std::optional<AxisAlignedBoundingBox> get_box();

    void set_box(float min_x, float max_x, float min_y, float max_y, float min_z, float max_z);

    std::optional<AxisAlignedBoundingBox> box;
};

