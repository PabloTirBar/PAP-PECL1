#include "blanco_negro.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void kernelBlancoNegro(byte* d_pixels, int width, int height, int bytesPerPixel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // columna
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // fila

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * bytesPerPixel;

    byte r = d_pixels[idx + 2];
    byte g = d_pixels[idx + 1];
    byte b = d_pixels[idx + 0];

    byte gris = static_cast<byte>(0.299f * r + 0.587f * g + 0.114f * b);

    d_pixels[idx + 0] = gris;
    d_pixels[idx + 1] = gris;
    d_pixels[idx + 2] = gris;
}

void convertirBlancoNegro(byte* h_pixels, int width, int height, int bytesPerPixel) {
    byte* d_pixels;
    size_t size = width * height * bytesPerPixel;

    cudaMalloc(&d_pixels, size);
    cudaMemcpy(d_pixels, h_pixels, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    kernelBlancoNegro << <gridSize, blockSize >> > (d_pixels, width, height, bytesPerPixel);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pixels, d_pixels, size, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}
