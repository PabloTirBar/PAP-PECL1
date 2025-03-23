#pragma once

#include "utils.h"

// Declaración del kernel pixelado con parámetro de tamaño constante
void pixelarImagen(
    byte* h_pixels,
    int width,
    int height,
    int bpp,
    int tamanoPixel
);