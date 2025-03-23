#include "pseudo_hash.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

#define THREADS_PER_BLOCK 512
#define MAX_HASH_SIZE 15

__device__ __forceinline__ int max_custom(int a, int b) {
    return a > b ? a : b;
}

__global__ void reducirPrimeraIteracion(byte* pixels, int width, int height, int bpp, int* salida) {
    __shared__ int sdata[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int total = width * height;

    int valor = 0;
    if (gid < total) {
        int idx = gid * bpp;
        byte r = pixels[idx + 2];
        byte g = pixels[idx + 1];
        byte b = pixels[idx + 0];
        valor = static_cast<int>(r * 0.5f + g * 0.25f + b * 0.25f);
    }

    sdata[tid] = valor;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid + s < total) {
            sdata[tid] = max_custom(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        salida[blockIdx.x] = sdata[0];
    }
}

__global__ void reducirSucesiva(int* entrada, int N, int* salida) {
    __shared__ int sdata[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (gid < N) ? entrada[gid] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid + s < N) {
            sdata[tid] = max_custom(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        salida[blockIdx.x] = sdata[0];
    }
}

void mostrarPseudoHash(byte* h_pixels, int width, int height, int bpp) {
    int total = width * height;
    int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    byte* d_pixels;
    int* d_intermedio;
    cudaMalloc(&d_pixels, total * bpp);
    cudaMemcpy(d_pixels, h_pixels, total * bpp, cudaMemcpyHostToDevice);

    cudaMalloc(&d_intermedio, blocks * sizeof(int));
    reducirPrimeraIteracion << <blocks, THREADS_PER_BLOCK >> > (d_pixels, width, height, bpp, d_intermedio);
    cudaDeviceSynchronize();

    std::vector<int> host_result(blocks);
    cudaMemcpy(host_result.data(), d_intermedio, blocks * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);
    cudaFree(d_intermedio);

    while (host_result.size() > MAX_HASH_SIZE) {
        int N = host_result.size();
        int nextBlocks = std::max(MAX_HASH_SIZE, (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

        int* d_in;
        int* d_out;
        cudaMalloc(&d_in, N * sizeof(int));
        cudaMalloc(&d_out, nextBlocks * sizeof(int));
        cudaMemcpy(d_in, host_result.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        reducirSucesiva << <nextBlocks, THREADS_PER_BLOCK >> > (d_in, N, d_out);
        cudaDeviceSynchronize();

        host_result.resize(nextBlocks);
        cudaMemcpy(host_result.data(), d_out, nextBlocks * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_in);
        cudaFree(d_out);
    }

    std::cout << "\nPseudo-hash generado:" << std::endl;

    std::cout << "Hash (original):       ";
    for (int val : host_result) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << "Hash (normalizado):    ";
    std::vector<int> normalizados;
    for (int val : host_result) {
        int norm = 35 + (val * (125 - 35)) / 255;
        normalizados.push_back(norm);
        std::cout << norm << " ";
    }
    std::cout << std::endl;

    std::cout << "Hash (ASCII):          ";
    for (int c : normalizados) std::cout << static_cast<char>(c) << " ";
    std::cout << std::endl;
}