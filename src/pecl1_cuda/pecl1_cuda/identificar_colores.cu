#include "identificar_colores.cuh"
#include <cuda_runtime.h>
#include <iostream>

__constant__ float c_umbral;
__constant__ float c_magnitud;

__global__ void kernelFiltrarColor(byte* d_pixels_in, byte* d_pixels_out,
    int width, int height, int bpp,
    int colorCode, int* d_count) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * bpp;

    byte b = d_pixels_in[idx + 0];
    byte g = d_pixels_in[idx + 1];
    byte r = d_pixels_in[idx + 2];

    bool esColor = false;

    if (colorCode == 0) { // ROJO
        esColor = (r >= 100 && r <= 255 && g < 150 && b < 150);
    }
    else if (colorCode == 1) { // VERDE
        esColor = (r > 30 && r <= 150 && g >= 50 && g <= 255 && b < 75);
    }
    else if (colorCode == 2) { // AZUL
        esColor = (r <= 200 && g < 250 && b >= 100 && b <= 255);
    }

    if (esColor) {
        d_pixels_out[idx + 0] = b;
        d_pixels_out[idx + 1] = g;
        d_pixels_out[idx + 2] = r;
        atomicAdd(d_count, 1);
    }
    else {
        byte gris = 255;
        d_pixels_out[idx + 0] = gris;
        d_pixels_out[idx + 1] = gris;
        d_pixels_out[idx + 2] = gris;
    }
}

void identificarColor(byte* h_pixels, int width, int height, int bytesPerPixel,
    ColorDetectado color, float umbral, float magnitud, const char* ruta_salida) {

    size_t size = width * height * bytesPerPixel;

    byte* d_pixels_in;
    byte* d_pixels_out;
    int* d_count;
    int h_count = 0;

    cudaMalloc(&d_pixels_in, size);
    cudaMalloc(&d_pixels_out, size);
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_pixels_in, h_pixels, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(c_umbral, &umbral, sizeof(float));
    cudaMemcpyToSymbol(c_magnitud, &magnitud, sizeof(float));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blockSize = 16;
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((width + blockSize - 1) / blockSize, (height + blockSize - 1) / blockSize);

    kernelFiltrarColor << <numBlocks, threadsPerBlock >> > (
        d_pixels_in, d_pixels_out, width, height, bytesPerPixel, (int)color, d_count);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pixels, d_pixels_out, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (!exportarBMP(ruta_salida, h_pixels, width, height, bytesPerPixel)) {
        std::cerr << "Error al guardar imagen en: " << ruta_salida << std::endl;
    }
    else {
        std::cout << "Imagen guardada en: " << ruta_salida << std::endl;
        std::cout << "Píxeles detectados: " << h_count << std::endl;
    }

    cudaFree(d_pixels_in);
    cudaFree(d_pixels_out);
    cudaFree(d_count);
}