#include "filtrar_delinear.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <string>

__constant__ float c_umbral;
__constant__ float c_magnitud;

__global__ void kernelFiltrarDibujar(byte* d_input, byte* d_output, int width, int height, int bpp, int colorCode) {
    extern __shared__ byte shared[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int idx = (y * width + x) * bpp;
    int localIdx = (ty * blockDim.x + tx) * bpp;

    if (x >= width || y >= height) return;

    shared[localIdx + 0] = d_input[idx + 0]; // B
    shared[localIdx + 1] = d_input[idx + 1]; // G
    shared[localIdx + 2] = d_input[idx + 2]; // R

    __syncthreads();

    byte b = shared[localIdx + 0];
    byte g = shared[localIdx + 1];
    byte r = shared[localIdx + 2];

    bool esColor = false;

    if (colorCode == 0) {
        esColor = (r >= 100 && r <= 255 && g < 150 && b < 150);
    }
    else if (colorCode == 1) {
        esColor = (r > 30 && r <= 150 && g >= 50 && g <= 255 && b < 75);
    }
    else if (colorCode == 2) {
        esColor = (r <= 200 && g < 250 && b >= 100 && b <= 255);
    }

    byte resultado[3] = { 200, 200, 200 };

    if (esColor) {
        bool borde = false;

        for (int dy = -1; dy <= 1 && !borde; dy++) {
            for (int dx = -1; dx <= 1 && !borde; dx++) {
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;

                int nIdx = (ny * width + nx) * bpp;
                byte nb = d_input[nIdx + 0];
                byte ng = d_input[nIdx + 1];
                byte nr = d_input[nIdx + 2];

                bool vecinoColor = false;
                if (colorCode == 0) vecinoColor = (nr >= 100 && nr <= 255 && ng < 150 && nb < 150);
                else if (colorCode == 1) vecinoColor = (nr > 30 && nr <= 150 && ng >= 50 && ng <= 255 && nb < 75);
                else if (colorCode == 2) vecinoColor = (nr <= 200 && ng < 250 && nb >= 100 && nb <= 255);

                if (!vecinoColor) borde = true;
            }
        }

        if (borde) {
            resultado[0] = 0;
            resultado[1] = 0;
            resultado[2] = 0;
        }
        else {
            resultado[0] = b;
            resultado[1] = g;
            resultado[2] = r;
        }
    }

    d_output[idx + 0] = resultado[0];
    d_output[idx + 1] = resultado[1];
    d_output[idx + 2] = resultado[2];
}

void filtrarYDelimitarColor(byte* h_pixels, int width, int height, int bpp, const char* colorNombre, float umbral, float magnitud) {
    size_t size = width * height * bpp;
    byte* d_input, * d_output;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_pixels, size, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(c_umbral, &umbral, sizeof(float));
    cudaMemcpyToSymbol(c_magnitud, &magnitud, sizeof(float));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blockSize = 16;
    dim3 threads(blockSize, blockSize);
    dim3 blocks((width + blockSize - 1) / blockSize, (height + blockSize - 1) / blockSize);

    int sharedMemSize = blockSize * blockSize * bpp;

    int colorCode = 0;
    if (strcmp(colorNombre, "verde") == 0) colorCode = 1;
    else if (strcmp(colorNombre, "azul") == 0) colorCode = 2;

    kernelFiltrarDibujar << <blocks, threads, sharedMemSize >> > (d_input, d_output, width, height, bpp, colorCode);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pixels, d_output, size, cudaMemcpyDeviceToHost);

    std::string ruta = std::string("C:\\Users\\iubal.camjalli\\Documents\\PAP\\PAP-PECL1\\imgs\\filtrar_delinear_") + colorNombre + ".bmp";

    if (!exportarBMP(ruta.c_str(), h_pixels, width, height, bpp)) {
        std::cerr << "Error al guardar imagen filtrada de color " << colorNombre << std::endl;
    }
    else {
        std::cout << "Imagen guardada: " << ruta << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}