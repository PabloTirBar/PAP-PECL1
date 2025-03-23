#ifndef FILTRAR_DELINEAR_CUH
#define FILTRAR_DELINEAR_CUH

#include "utils.h"

void filtrarYDelimitarColor(byte* h_pixels, int width, int height, int bpp, const char* colorNombre, float umbral, float magnitud);

#endif