#pragma once

#include "utils.h"
#include "identificar_colores.cuh"  // Reutilizar enum existente

// Función principal para filtro y delineado de color
void filtrarYDelimitarColor(
    byte* h_pixels,
    int width,
    int height,
    int bpp,
    ColorDetectado color,
    float umbral,
    int halo,
    const char* ruta_salida
);