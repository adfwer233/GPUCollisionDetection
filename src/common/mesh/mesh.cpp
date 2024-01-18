#include "common/mesh/mesh.hpp"

#include "GLFW/glfw3.h"
#include "glad/glad.h"
#include "glm/gtc/type_ptr.hpp"

std::optional<AxisAlignedBoundingBox> Mesh::get_box() {
    return this->box;
}

void Mesh::set_box(float min_x, float max_x, float min_y, float max_y, float min_z, float max_z) {
    this->box.value() = {min_x, max_x, min_y, max_y, min_z, max_z};
}

void Mesh::bind_buffer() {
    glGenBuffers(1, &VBO);
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, vertices.size() * (sizeof(decltype(vertices)::value_type)), vertices.data(),
                 GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(decltype(vertices)::value_type), (void *)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(decltype(vertices)::value_type),
                          (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(decltype(vertices)::value_type),
                          (void *)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, face_indices.size() * sizeof(TriangleVerticeIndex), face_indices.data(),
                 GL_STATIC_DRAW);
}

void Mesh::process_rendering(Shader &shader, Camera camera, glm::vec3 lightPos) {
    auto projection = glm::perspective(glm::radians(camera.zoom), 1.0f, 0.1f, 100.0f);

    shader.use();
    shader.set_vec3("objectColor", this->object_color);
    shader.set_vec3("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
    shader.set_vec3("lightPos", lightPos);
    shader.set_vec3("viewPos", camera.position);

    int transformLoc = glGetUniformLocation(shader.ID, "model");
    glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm::value_ptr(this->transform));

    int viewTransformLoc = glGetUniformLocation(shader.ID, "view");
    glUniformMatrix4fv(viewTransformLoc, 1, GL_FALSE, glm::value_ptr(camera.get_view_transformation()));

    int projectionTransformLoc = glGetUniformLocation(shader.ID, "projection");
    glUniformMatrix4fv(projectionTransformLoc, 1, GL_FALSE, glm::value_ptr(projection));

    glBindVertexArray(this->VAO);

    if (this->strip_flag) {
        glDrawElements(GL_LINE_STRIP, face_indices.size() * 3, GL_UNSIGNED_INT, nullptr);
    } else {
        glDrawElements(GL_TRIANGLES, face_indices.size() * 3, GL_UNSIGNED_INT, nullptr);
    }

    glBindVertexArray(0);
}

void Mesh::process_instanced_rendering(Mesh &mesh, size_t amount, Shader &shader, Camera &camera, glm::vec3 lightPos) {
    auto projection = glm::perspective(glm::radians(camera.zoom), 1.0f, 0.1f, 100.0f);

    shader.use();
    shader.set_vec3("objectColor", {1.0, 0, 0});
    shader.set_vec3("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
    shader.set_vec3("lightPos", lightPos);
    shader.set_vec3("viewPos", camera.position);

    int viewTransformLoc = glGetUniformLocation(shader.ID, "view");
    glUniformMatrix4fv(viewTransformLoc, 1, GL_FALSE, glm::value_ptr(camera.get_view_transformation()));

    int projectionTransformLoc = glGetUniformLocation(shader.ID, "projection");
    glUniformMatrix4fv(projectionTransformLoc, 1, GL_FALSE, glm::value_ptr(projection));

    glBindVertexArray(mesh.VAO);
    glDrawElementsInstanced(GL_TRIANGLES, mesh.face_indices.size() * 3, GL_UNSIGNED_INT, nullptr, amount);
    glBindVertexArray(0);
}
