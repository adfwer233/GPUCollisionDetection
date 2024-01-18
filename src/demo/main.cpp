#include <format>

#include "collision_detection/cpu/cpu_collision_detection.hpp"
#include "collision_detection/gpu/gpu_collision_detection.hpp"

#include "common/simulation/field.hpp"
#include "common/simulation/simulation.hpp"
#include "common/solid/solid.hpp"
#include "iostream"
#include "random"

#include "GLFW/glfw3.h"
#include "glad/glad.h"

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 1024;
const unsigned int SCR_HEIGHT = 1024;

#ifndef SHADER_DIR
#define SHADER_DIR "./shader"
#endif

Camera camera(glm::vec3(0, 0, 10.0f), glm::vec3(0, 1.0f, 0));
glm::vec3 lightPos(0, 0, 10);

float deltaTime = 0.0f;
float lastFrame = 0.0f;
void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.process_keyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.process_keyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.process_keyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.process_keyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camera.process_keyboard(UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        camera.process_keyboard(DOWN, deltaTime);
}

void scroll_callback(GLFWwindow *window, double x_offset, double y_offset) {
    camera.process_mouse_scroll(y_offset);
}

bool is_mouse_pressing = false;
bool mouse_flag = true;

void mouse_button_callback(GLFWwindow *window, int button, int state, int mod) {
    if (button == GLFW_MOUSE_BUTTON_LEFT and state == GLFW_PRESS)
        is_mouse_pressing = true;
    if (button == GLFW_MOUSE_BUTTON_LEFT and state == GLFW_RELEASE) {
        mouse_flag = true;
        is_mouse_pressing = false;
    }
}

float last_x = SCR_WIDTH / 2.0f;
float last_y = SCR_HEIGHT / 2.0f;

void mouse_callback(GLFWwindow *window, double xposIn, double yposIn) {
    if (not is_mouse_pressing)
        return;

    float x_pos = static_cast<float>(xposIn);
    float y_pos = static_cast<float>(yposIn);

    if (mouse_flag) {
        last_x = x_pos;
        last_y = y_pos;
        mouse_flag = false;
    }

    float x_offset = x_pos - last_x;
    float y_offset = last_y - y_pos; // reversed since y-coordinates go from bottom to top

    last_x = x_pos;
    last_y = y_pos;

    camera.process_mouse_movement(x_offset, y_offset);
}

int main() {
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Collision Detection Demo", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    Shader common_shader(std::format("{}/common.vs", SHADER_DIR), std::format("{}/common.fs", SHADER_DIR));

    Shader instanced_shader(std::format("{}/instanced.vs", SHADER_DIR), std::format("{}/instanced.fs", SHADER_DIR));

    std::vector<std::reference_wrapper<Solid>> solid_vector;

    Ball ball({-1, 0, 0}, 1, {0, 0, 0}, 0.2);
    Ball ball2({-1, 0, 1}, 1, {0, 0, 0}, 0.2);
    Ball ball3({-1, 0.5, 0}, 1, {0, 1, 0}, 0.2);
    Ball ball4({-1, 0.5, 1}, 1, {0, 1, 0}, 0.2);

    std::vector<Ball> balls;

    int x_num = 100;
    int z_num = 100;

    for (int i = 0; i < x_num; i++) {
        for (int j = 0; j < z_num; j++) {
            float delta_x = 20.0 / x_num * i;
            float delta_z = 20.0 / z_num * j;
            balls.emplace_back(glm::vec3{-10 + delta_x, 0, -10 + delta_z}, 1, glm::vec3{0, 0, 0}, 0.02);
            balls.emplace_back(glm::vec3{-10 + delta_x, 0.5, -10 + delta_z}, 1, glm::vec3{0, 2, 0}, 0.02);
        }
    }

    for (auto &ball : balls)
        solid_vector.push_back(ball);

    std::random_device seed;
    std::ranlux48 engine(seed());
    std::uniform_real_distribution distrib(0.0f, 1.0f);

    //  std::vector<Mesh> mesh_vector;
    //  mesh_vector.reserve(solid_vector.size());

    Mesh ball_mesh = solid_vector[0].get().construct_mesh();

    std::vector<glm::mat4> model_matrices(solid_vector.size());
    std::vector<glm::vec3> object_color(solid_vector.size());

    for (int i = 0; auto solid : solid_vector) {
        object_color[i] = {distrib(engine), distrib(engine), distrib(engine)};
        solid.get().mesh_ref = ball_mesh;
        i++;
    }

    GravityField field;

    Ground ground({0, -5, 0});
    Mesh ground_mesh = ground.construct_mesh();
    ground_mesh.bind_buffer();

    ground_mesh.object_color = {0.5, 0.5, 0};

    GPUSweepAndPruneCollisionDetection collision_detector(solid_vector.size());

    for (size_t i = 0; i < solid_vector.size(); i++) {
        model_matrices[i] = glm::translate(glm::mat4(1.0f), solid_vector[i].get().center);
    }

    ball_mesh.bind_buffer();
    ball_mesh.transform = glm::mat4(1.0f);

    unsigned int instance_transform_buffer;
    glGenBuffers(1, &instance_transform_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, instance_transform_buffer);
    glBufferData(GL_ARRAY_BUFFER, solid_vector.size() * sizeof(glm::mat4), model_matrices.data(), GL_STATIC_DRAW);

    glBindVertexArray(ball_mesh.VAO);

    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)0);
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(sizeof(glm::vec4)));
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(2 * sizeof(glm::vec4)));
    glEnableVertexAttribArray(7);
    glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(3 * sizeof(glm::vec4)));

    glVertexAttribDivisor(4, 1);
    glVertexAttribDivisor(5, 1);
    glVertexAttribDivisor(6, 1);
    glVertexAttribDivisor(7, 1);

    glBindVertexArray(0);


    unsigned int instance_object_color_buffer;
    glGenBuffers(1, &instance_object_color_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, instance_object_color_buffer);
    glBufferData(GL_ARRAY_BUFFER, solid_vector.size() * sizeof(glm::vec3), object_color.data(), GL_STATIC_DRAW);

    glBindVertexArray(ball_mesh.VAO);

    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

    glVertexAttribDivisor(3, 1);
    glBindVertexArray(0);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        std::cout << 1.0f / deltaTime << std::endl;

        Simulator simulator(0.005);

        for (size_t i = 0; i < solid_vector.size(); i++) {
            auto displacement = simulator.update_state_with_field(solid_vector[i], field);
            model_matrices[i] = glm::translate(model_matrices[i], displacement);
        }

        glBindBuffer(GL_ARRAY_BUFFER, instance_transform_buffer);
        glBufferData(GL_ARRAY_BUFFER, solid_vector.size() * sizeof(glm::mat4), model_matrices.data(), GL_STATIC_DRAW);

        // collision_detection between solids

        //         CPUNaiveCollisionDetection::collision_detection(solid_vector);

        //    CPUSweepAndPruneCollisionDetection::collision_detection(solid_vector,0.005);

        collision_detector.collision_detection(solid_vector, 0.005);

        // collision detection between solid and ground

        for (auto &i : solid_vector) {
            bool res = simulator.update_state_when_collide(ground, i);
        }

        processInput(window);

        // render
        // ------
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //    for (auto& mesh : mesh_vector) {
        //      mesh.process_rendering(shader, camera, lightPos);
        //    }

        Mesh::process_instanced_rendering(ball_mesh, solid_vector.size(), instanced_shader, camera, lightPos);

        ground_mesh.process_rendering(common_shader, camera, lightPos);
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved
        // etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// glfw: whenever the window size changed (by OS or user resize) this callback
// function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    // make sure the viewport matches the new window dimensions; note that width
    // and height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}