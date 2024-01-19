#pragma once

#include "common/camera/camera.hpp"
#include "common/shader/shader.hpp"
#include "glm/glm.hpp"
#include "optional"
#include "vector"

/**
 * @brief AABB box struct
 */
struct AxisAlignedBoundingBox {
    float min_x, max_x, min_y, max_y, min_z, max_z;
};

/**
 * @brief struct used in OpenGL EBO
 */
struct TriangleVerticeIndex {
    unsigned int x, y, z;
};

/**
 * @brief struct used in OpenGL VBO
 */
struct TriangleWithNormal {
    glm::vec3 point, normal;
    glm::vec2 texture_coord;
};

/**
 * @brief Mesh for rendering
 */
class Mesh {
  public:
    /**
     * @brief vertices
     */
    std::vector<TriangleWithNormal> vertices;

    /**
     * @brief indices of triangles
     */
    std::vector<TriangleVerticeIndex> face_indices;

    /**
     * @brief model transformation
     */
    glm::mat4 transform;

    /**
     * @brief object color
     */
    glm::vec3 object_color{1, 1, 1};

    /**
     * @brief OpenGL Buffers
     */
    unsigned int VAO{}, VBO{}, EBO{};

    /**
     * @brief if visualized in strip mode
     */
    bool strip_flag{false};

    /**
     * @brief bind opengl buffer
     */
    void bind_buffer();

    /**
     * @brief Render this mesh
     * @param shader
     * @param camera
     * @param lightPos
     */
    void process_rendering(Shader &shader, Camera camera, glm::vec3 lightPos);

    /**
     * @brief get bounding box of this mesh
     * @return optional aabb bounding box
     */
    std::optional<AxisAlignedBoundingBox> get_box();

    /**
     * @brief set mesh box
     * @param min_x
     * @param max_x
     * @param min_y
     * @param max_y
     * @param min_z
     * @param max_z
     */
    void set_box(float min_x, float max_x, float min_y, float max_y, float min_z, float max_z);

    /**
     * @brief mesh bounding box
     */
    std::optional<AxisAlignedBoundingBox> box;

    /**
     * @brief Static method for OpenGL instanced rendering
     *
     * When we want to draw many instances of same mesh, use instanced rendering can significantly reduce the
     * number of draw calls.
     *
     * @param mesh mesh to perform instanced rendering
     * @param amount number of instances.
     * @param shader
     * @param camera
     * @param lightPos
     */
    static void process_instanced_rendering(Mesh &mesh, size_t amount, Shader &shader, Camera &camera,
                                            glm::vec3 lightPos);
};
