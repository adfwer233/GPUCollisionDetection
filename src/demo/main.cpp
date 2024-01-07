#include <format>
#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "iostream"

#include "random"

#include "common/simulation/simulation.hpp"
#include "common/simulation/field.hpp"
#include "common/solid/solid.hpp"

#include "collision_detection/cpu/cpu_collision_detection.hpp"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
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
void processInput(GLFWwindow* window) {

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

void scroll_callback(GLFWwindow* window, double x_offset, double y_offset) {
    camera.process_mouse_scroll(y_offset);
}

bool is_mouse_pressing = false;
bool mouse_flag = true;

void mouse_button_callback(GLFWwindow* window, int button, int state, int mod) {
    if (button == GLFW_MOUSE_BUTTON_LEFT and state == GLFW_PRESS)
        is_mouse_pressing = true;
    if (button == GLFW_MOUSE_BUTTON_LEFT and state == GLFW_RELEASE) {
        mouse_flag = true;
        is_mouse_pressing = false;
    }
}


float last_x = SCR_WIDTH / 2.0f;
float last_y = SCR_HEIGHT / 2.0f;

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    if (not is_mouse_pressing) return;

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

int main()
{
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
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Collision Detection Demo", NULL, NULL);
    if (window == NULL)
    {
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
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    Shader shader(std::format("{}/common.vs", SHADER_DIR), std::format("{}/common.fs", SHADER_DIR));

    std::vector<std::reference_wrapper<Solid>> solid_vector;
    std::vector<Mesh> mesh_vector;

    Ball ball({-1, 0 ,0}, 1, {0, 0, 0}, 0.2);
    Ball ball2({-1, 0 ,1}, 1, {0, 0, 0}, 0.2);
    Ball ball3({-1, 0.5 ,0}, 1, {0, 1, 0}, 0.2);
    Ball ball4({-1, 0.5 ,1}, 1, {0, 1, 0}, 0.2);

    solid_vector.push_back(ball);
    solid_vector.push_back(ball2);
    solid_vector.push_back(ball3);
    solid_vector.push_back(ball4);

    std::random_device seed;
    std::ranlux48 engine(seed());
    std::uniform_real_distribution distrib(0.0f, 1.0f);

    for (auto solid: solid_vector) {
        mesh_vector.push_back(solid.get().construct_mesh());
        Mesh &mesh = mesh_vector.back();
        mesh.bind_buffer();
        mesh.object_color = {distrib(engine), distrib(engine), distrib(engine)};
    }

    GravityField field;

    Ground ground({0, -5, 0});
    Mesh ground_mesh = ground.construct_mesh();
    ground_mesh.bind_buffer();

    ground_mesh.object_color = {0.5, 0.5, 0};

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        Simulator simulator(deltaTime);

        for (size_t i = 0; i < solid_vector.size(); i++) {
            auto displacement = simulator.update_state_with_field(solid_vector[i], field);
            Mesh &mesh = mesh_vector[i];
            mesh.transform = glm::translate(mesh.transform, displacement);
        }

        // collision_detection between solids

        CPUNaiveCollisionDetection::collision_detection(solid_vector);

        // collision detection between solid and ground

        for (auto &i : solid_vector) {
            bool res = simulator.update_state_when_collide(ground, i);
        }

        processInput(window);

        // render
        // ------
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        for (auto &mesh: mesh_vector) {
            mesh.process_rendering(shader, camera, lightPos);
        }

        ground_mesh.process_rendering(shader, camera, lightPos);
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}