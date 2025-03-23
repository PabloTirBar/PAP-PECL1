#pragma once
#include "utils.h"
#include "identificar_colores.cuh"


void filtrarYDelimitarColor(byte* h_pixels, int width, int height, int bpp,
    ColorDetectado color, float umbral, float magnitud,
    const char* ruta_salida);