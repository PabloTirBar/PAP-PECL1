#ifndef UTILS_H
#define UTILS_H

#include <cstdint>  // Para uint8_t, int32_t

typedef unsigned int int32;
typedef short int16;
typedef unsigned char byte;

// Funciones para manejar imágenes BMP
bool cargarBMP(const char* fileName, byte** pixels, int32* width, int32* height, int32* bytesPerPixel);
bool exportarBMP(const char* fileName, byte* pixels, int32 width, int32 height, int32 bytesPerPixel);

#endif