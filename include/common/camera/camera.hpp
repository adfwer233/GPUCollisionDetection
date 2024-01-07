#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

constexpr float default_pitch = 0.0f;
constexpr float default_yaw = -90.0f;

enum CameraMovement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    DOWN,
    UP
};

class Camera {
private:
    // calculates the front vector from the Camera's (updated) Euler Angles
    void update_camera_vectors();
public:
    // position of camera in world coordinate
    glm::vec3 position;

    // position of scene origin in world space
    glm::vec3 camera_target;

    // up direction of camera (normalized)
    glm::vec3 camera_up_axis;

    // right axis direction of camera (normalized)
    glm::vec3 camera_right_axis;

    // camera front
    glm::vec3 camera_front;

    glm::vec3 world_up;

    // euler angle
    float yaw;
    float pitch;

    // camera options
    float zoom;
    const float move_speed;
    const float mouse_sensitivity;

    Camera(glm::vec3 pos, glm::vec3 up, float t_yaw = default_yaw, float t_pitch = default_pitch): zoom(45), move_speed(2.5), mouse_sensitivity(0.1f) {
        position = pos;
        camera_up_axis = up;
        yaw = t_yaw;
        pitch = t_pitch;
        world_up = up;
        update_camera_vectors();
    }

    glm::mat4 get_view_transformation() const;

    void process_mouse_scroll(float offset);

    void process_mouse_movement(float x_offset, float y_offset);

    void process_keyboard(CameraMovement direction, float deltaTime);
};