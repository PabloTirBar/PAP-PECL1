#include "pixelado.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void kernelPixelar(byte* d_pixels, int width, int height, int bytesPerPixel, int blockSize) {
    int bx = blockIdx.x * blockSize;
    int by = blockIdx.y * blockSize;

    if (bx >= width || by >= height) return;

    int sumR = 0, sumG = 0, sumB = 0;
    int count = 0;

    for (int i = 0; i < blockSize && (by + i) < height; ++i) {
        for (int j = 0; j < blockSize && (bx + j) < width; ++j) {
            int idx = ((by + i) * width + (bx + j)) * bytesPerPixel;
            sumR += d_pixels[idx + 2];
            sumG += d_pixels[idx + 1];
            sumB += d_pixels[idx + 0];
            count++;
        }
    }

    byte avgR = sumR / count;
    byte avgG = sumG / count;
    byte avgB = sumB / count;

    for (int i = 0; i < blockSize && (by + i) < height; ++i) {
        for (int j = 0; j < blockSize && (bx + j) < width; ++j) {
            int idx = ((by + i) * width + (bx + j)) * bytesPerPixel;
            d_pixels[idx + 0] = avgB;
            d_pixels[idx + 1] = avgG;
            d_pixels[idx + 2] = avgR;
        }
    }
}

void pixelarImagen(byte* h_pixels, int width, int height, int bytesPerPixel, int blockSize) {
    byte* d_pixels;
    size_t size = width * height * bytesPerPixel;

    cudaMalloc(&d_pixels, size);
    cudaMemcpy(d_pixels, h_pixels, size, cudaMemcpyHostToDevice);

    dim3 gridSize((width + blockSize - 1) / blockSize, (height + blockSize - 1) / blockSize);

    kernelPixelar << <gridSize, 1 >> > (d_pixels, width, height, bytesPerPixel, blockSize);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pixels, d_pixels, size, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}