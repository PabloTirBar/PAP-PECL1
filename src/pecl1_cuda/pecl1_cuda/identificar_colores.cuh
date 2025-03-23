#ifndef IDENTIFICAR_COLORES_CUH
#define IDENTIFICAR_COLORES_CUH

#include "utils.h"


enum ColorDetectado { ROJO = 0, VERDE = 1, AZUL = 2 };

// Firma de la función (con ruta como string):
void identificarColor(byte* h_pixels, int width, int height, int bytesPerPixel,
    ColorDetectado color, float umbral,
    const char* rutaSalida);

#endif