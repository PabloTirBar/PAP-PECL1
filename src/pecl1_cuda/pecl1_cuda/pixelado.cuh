#pragma once

#include "utils.h"

// Declaraci�n del kernel pixelado con par�metro de tama�o constante
void pixelarImagen(
    byte* h_pixels,
    int width,
    int height,
    int bpp,
    int tamanoPixel
);