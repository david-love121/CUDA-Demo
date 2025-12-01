#include <iostream>
#include <vector>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include "renderer.h"

const int WIDTH = 800;
const int HEIGHT = 600;

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "CUDA Rotating Cube", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // Allocate CUDA buffer
    uint32_t* d_buffer;
    cudaMalloc(&d_buffer, WIDTH * HEIGHT * sizeof(uint32_t));

    // Host buffer for texture upload
    std::vector<uint32_t> h_buffer(WIDTH * HEIGHT);

    // Create OpenGL Texture
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    while (!glfwWindowShouldClose(window)) {
        float time = (float)glfwGetTime();

        // Render on GPU
        render_frame(d_buffer, WIDTH, HEIGHT, time);

        // Copy back to CPU (Slow, but simple for this demo)
        cudaMemcpy(h_buffer.data(), d_buffer, WIDTH * HEIGHT * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Upload to Texture
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_buffer.data());

        // Draw Fullscreen Quad
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);

        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f); // Bottom Left (OpenGL origin is bottom-left, but image is top-down usually?)
        // Raymarcher: (0,0) is center.
        // Buffer: Row major.
        // OpenGL Texture: (0,0) is bottom left.
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f); // Bottom Right
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f); // Top Right
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f); // Top Left
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaFree(d_buffer);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
