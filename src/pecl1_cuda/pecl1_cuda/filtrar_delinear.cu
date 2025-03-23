#include "filtrar_delinear.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include "identificar_colores.cuh"

__constant__ float c_umbral;
__constant__ int c_halo;

__global__ void kernelFiltrarDibujar(byte* d_input, byte* d_output, int width, int height, int bpp, int colorCode) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * bpp;

    byte b = d_input[idx + 0];
    byte g = d_input[idx + 1];
    byte r = d_input[idx + 2];

    bool esColor = false;

    if (colorCode == 0) {
        esColor = (r >= 100 - c_umbral && r <= 255 + c_umbral &&
            g >= 0 - c_umbral && g < 150 + c_umbral &&
            b >= 0 - c_umbral && b < 150 + c_umbral);
    }
    else if (colorCode == 1) {
        esColor = (r > 30 - c_umbral && r <= 150 + c_umbral &&
            g >= 50 - c_umbral && g <= 255 + c_umbral &&
            b >= 0 - c_umbral && b < 75 + c_umbral);
    }
    else if (colorCode == 2) {
        esColor = (r >= 0 - c_umbral && r <= 200 + c_umbral &&
            g >= 0 - c_umbral && g < 250 + c_umbral &&
            b >= 100 - c_umbral && b <= 255 + c_umbral);
    }

    if (esColor) {
        d_output[idx + 0] = b;
        d_output[idx + 1] = g;
        d_output[idx + 2] = r;
        return;
    }

    // Ver si está cerca de una región de color -> halo negro
    bool cerca = false;
    for (int dy = -c_halo; dy <= c_halo && !cerca; dy++) {
        for (int dx = -c_halo; dx <= c_halo && !cerca; dx++) {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;

            int nIdx = (ny * width + nx) * bpp;
            byte nb = d_input[nIdx + 0];
            byte ng = d_input[nIdx + 1];
            byte nr = d_input[nIdx + 2];

            bool vecinoColor = false;
            if (colorCode == 0)
                vecinoColor = (nr >= 100 - c_umbral && nr <= 255 + c_umbral &&
                    ng >= 0 - c_umbral && ng < 150 + c_umbral &&
                    nb >= 0 - c_umbral && nb < 150 + c_umbral);
            else if (colorCode == 1)
                vecinoColor = (nr > 30 - c_umbral && nr <= 150 + c_umbral &&
                    ng >= 50 - c_umbral && ng <= 255 + c_umbral &&
                    nb >= 0 - c_umbral && nb < 75 + c_umbral);
            else if (colorCode == 2)
                vecinoColor = (nr >= 0 - c_umbral && nr <= 200 + c_umbral &&
                    ng >= 0 - c_umbral && ng < 250 + c_umbral &&
                    nb >= 100 - c_umbral && nb <= 255 + c_umbral);

            if (vecinoColor) cerca = true;
        }
    }

    if (cerca) {
        d_output[idx + 0] = 0;
        d_output[idx + 1] = 0;
        d_output[idx + 2] = 0;
    }
    else {
        byte gris = (byte)(0.299f * r + 0.587f * g + 0.114f * b);
        d_output[idx + 0] = gris;
        d_output[idx + 1] = gris;
        d_output[idx + 2] = gris;
    }
}

void filtrarYDelimitarColor(byte* h_pixels, int width, int height, int bpp,
    ColorDetectado color, float umbral, int halo, const char* ruta_salida) {

    size_t size = width * height * bpp;
    byte* d_input;
    byte* d_output;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_pixels, size, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(c_umbral, &umbral, sizeof(float));
    cudaMemcpyToSymbol(c_halo, &halo, sizeof(int));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blockSize = 16;
    dim3 threads(blockSize, blockSize);
    dim3 blocks((width + blockSize - 1) / blockSize, (height + blockSize - 1) / blockSize);

    int colorCode = static_cast<int>(color);

    kernelFiltrarDibujar << <blocks, threads >> > (
        d_input, d_output, width, height, bpp, colorCode);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pixels, d_output, size, cudaMemcpyDeviceToHost);

    if (!exportarBMP(ruta_salida, h_pixels, width, height, bpp)) {
        std::cerr << "Error al guardar imagen filtrada de color en: " << ruta_salida << std::endl;
    }
    else {
        std::cout << "Imagen guardada en: " << ruta_salida << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}