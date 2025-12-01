#include <cmath>
#include "renderer.h"
#include <cuda_runtime.h>

__device__ float3 make_float3(float s) { return make_float3(s, s, s); }
__device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float3 normalize(float3 v) { float len = sqrtf(dot(v, v)); return v * (1.0f / len); }

// Rotate vector p around axis (0,1,0) by angle
__device__ float3 rotateY(float3 p, float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    return make_float3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

// Rotate vector p around axis (1,0,0) by angle
__device__ float3 rotateX(float3 p, float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    return make_float3(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
}

// Signed Distance Function for a Box. 
// If p is outside of b, then return positive value.
// If p is inside of b, then return negative value.
// Iff p is exactly on the edge, return 0.
__device__ float sdBox(float3 p, float3 b) {
    //Adjust box so corner is on the origin
    float3 q = make_float3(fabsf(p.x), fabsf(p.y), fabsf(p.z)) - b;
    //outside distance: if p is within the box (inside) then q will be negative and m will be 0.
    float3 m = make_float3(fmaxf(q.x, 0.0f), fmaxf(q.y, 0.0f), fmaxf(q.z, 0.0f));

    return sqrtf(dot(m, m)) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f);
}

// Signed Distance Function for a Sphere
__device__ float sdSphere(float3 p, float s) {
    return sqrtf(dot(p, p)) - s;
}

__device__ float map(float3 p, float time) {
    // Rotate the space to rotate the cube
    p = rotateY(p, time);
    p = rotateX(p, time * 0.5f);
    float3 object2Pos = make_float3(0.0f, 0.0f, 0.0f);
    float3 localPCoordinate = p - object2Pos;
    float3 Rectangle = make_float3(0.5f, 1.0f, 0.5f);
    float3 Cube = make_float3(0.5);
    return sdBox(localPCoordinate, Cube);
}

//Normal vector can be obtained by taking infinitesimal step to either direction along the surface of the object
//Finds the gradient of the surface 
__device__ float3 getNormal(float3 p, float time) {
    float e = 0.001f;
    return normalize(make_float3(
        map(p + make_float3(e, 0, 0), time) - map(p - make_float3(e, 0, 0), time),
        map(p + make_float3(0, e, 0), time) - map(p - make_float3(0, e, 0), time),
        map(p + make_float3(0, 0, e), time) - map(p - make_float3(0, 0, e), time)
    ));
}

__global__ void render_kernel(uint32_t* buffer, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // UV coordinates [-1, 1]
    float2 uv = make_float2((float)x / width * 2.0f - 1.0f, (float)y / height * 2.0f - 1.0f);
    uv.y *= (float)height / width; // Aspect ratio correction

    // Ray setup
    float3 ro = make_float3(0.0f, 0.0f, -3.0f); // Ray origin
    float3 rd = normalize(make_float3(uv.x, uv.y, 1.0f)); // Ray direction

    // Raymarching
    float t = 0.0f; // Total distance traveled along ray
    float d = 0.0f; // Distance to nearest object at ray
    int i;
    //Try to find an object 64 times
    for (i = 0; i < 64; i++) {
        float3 p = ro + rd * t; // p = cartesian coordinates where the ray is
        d = map(p, time); // How far is the object from our ray?
        if (d < 0.001f || t > 10.0f) break; //Found the object or ran off into the distance, break
        t += d;
    }

    // Coloring
    float3 col = make_float3(0.1f, 0.1f, 0.15f); // Background
    if (d < 0.001f) {
        float3 p = ro + rd * t;
        float3 n = getNormal(p, time);
        float3 lightDir = normalize(make_float3(0.5f, 1.0f, -0.5f));
        float diff = fmaxf(dot(n, lightDir), 0.0f);
        col = make_float3(1.0f, 0.8f, 0.1f) * diff + make_float3(0.1f, 0.1f, 0.2f);
    }

    // Gamma correction
    col.x = powf(col.x, 0.4545f);
    col.y = powf(col.y, 0.4545f);
    col.z = powf(col.z, 0.4545f);

    // Pack to RGBA8
    uint32_t r = (uint32_t)(fminf(col.x, 1.0f) * 255.0f);
    uint32_t g = (uint32_t)(fminf(col.y, 1.0f) * 255.0f);
    uint32_t b = (uint32_t)(fminf(col.z, 1.0f) * 255.0f);
    
    buffer[y * width + x] = 0xFF000000 | (b << 16) | (g << 8) | r;
}

void render_frame(uint32_t* d_buffer, int width, int height, float time) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    render_kernel<<<gridSize, blockSize>>>(d_buffer, width, height, time);
    cudaDeviceSynchronize();
}
