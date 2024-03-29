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

/**
 * @brief Camera class for opengl
 */
class Camera {
  private:
    /**
     * @brief calculates the front vector from the Camera's (updated) Euler Angles
     */
    void update_camera_vectors();

  public:
    /**
     * @brief position of camera in world coordinate
     */
    glm::vec3 position;

    /**
     * @brief position of scene origin in world space
     */
    glm::vec3 camera_target;

    /**
     * @brief up direction of camera (normalized)
     */
    glm::vec3 camera_up_axis;

    /**
     * @brief right axis direction of camera (normalized)
     */
    glm::vec3 camera_right_axis;

    /**
     * @brief front vector of camera
     */
    glm::vec3 camera_front;

    /**
     * @brief world up
     */
    glm::vec3 world_up;

    /**
     * @brief euler angle: yaw
     */
    float yaw;

    /**
     * @brief euler angle: pitch
     */
    float pitch;

    /**
     * @brief camera zoom
     */
    float zoom;

    /**
     * @brief move speed for interaction
     */
    const float move_speed;

    /**
     * @brief mouse sensitivity
     */
    const float mouse_sensitivity;

    Camera(glm::vec3 pos, glm::vec3 up, float t_yaw = default_yaw, float t_pitch = default_pitch)
        : zoom(45), move_speed(2.5), mouse_sensitivity(0.1f) {
        position = pos;
        camera_up_axis = up;
        yaw = t_yaw;
        pitch = t_pitch;
        world_up = up;
        update_camera_vectors();
    }

    glm::mat4 get_view_transformation() const;

    /**
     * @brief update camera state with an offset given by mouse scroll
     * @param offset
     */
    void process_mouse_scroll(float offset);

    /**
     * @brief update camera state with offsets given by mouse movement
     * @param x_offset
     * @param y_offset
     */
    void process_mouse_movement(float x_offset, float y_offset);

    /**
     * @brief process keyboard with given direction and delta time
     * @param direction
     * @param deltaTime
     */
    void process_keyboard(CameraMovement direction, float deltaTime);
};