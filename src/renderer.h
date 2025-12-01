#pragma once
#include <cstdint>

// Renders a frame to the given buffer.
// buffer: Device pointer to RGBA8 pixel data (width * height * 4 bytes).
// width: Width of the screen.
// height: Height of the screen.
// time: Time in seconds for animation.
void render_frame(uint32_t* d_buffer, int width, int height, float time);
