#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESSION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0

typedef unsigned int int32;
typedef short int int16;
typedef unsigned char byte;

bool cargarBMP(const char* ruta, byte** pixels, int32* width, int32* height, int32* bytesPerPixel) {
    FILE* imageFile = fopen(ruta, "rb");
    if (!imageFile) {
        printf("Error: No se pudo abrir el archivo.\n");
        return false;
    }

    int32 dataOffset;
    fseek(imageFile, DATA_OFFSET_OFFSET, SEEK_SET);
    fread(&dataOffset, 4, 1, imageFile);
    fseek(imageFile, WIDTH_OFFSET, SEEK_SET);
    fread(width, 4, 1, imageFile);
    fseek(imageFile, HEIGHT_OFFSET, SEEK_SET);
    fread(height, 4, 1, imageFile);
    int16 bitsPerPixel;
    fseek(imageFile, BITS_PER_PIXEL_OFFSET, SEEK_SET);
    fread(&bitsPerPixel, 2, 1, imageFile);
    *bytesPerPixel = bitsPerPixel / 8;

    int paddedRowSize = (int)(4 * ceil((float)(*width) / 4.0f)) * (*bytesPerPixel);
    int unpaddedRowSize = (*width) * (*bytesPerPixel);
    int totalSize = unpaddedRowSize * (*height);
    *pixels = (byte*)malloc(totalSize);

    byte* currentRowPointer = *pixels + ((*height - 1) * unpaddedRowSize);
    for (int i = 0; i < *height; i++) {
        fseek(imageFile, dataOffset + (i * paddedRowSize), SEEK_SET);
        fread(currentRowPointer, 1, unpaddedRowSize, imageFile);
        currentRowPointer -= unpaddedRowSize;
    }

    fclose(imageFile);
    return true;
}

bool exportarBMP(const char* ruta, byte* pixels, int32 width, int32 height, int32 bytesPerPixel) {
    FILE* outputFile = fopen(ruta, "wb");
    if (!outputFile) {
        printf("Error: No se pudo crear el archivo de salida.\n");
        return false;
    }

    const char* BM = "BM";
    fwrite(&BM[0], 1, 1, outputFile);
    fwrite(&BM[1], 1, 1, outputFile);

    int paddedRowSize = (int)(4 * ceil((float)width / 4.0f)) * bytesPerPixel;
    int32 fileSize = paddedRowSize * height + HEADER_SIZE + INFO_HEADER_SIZE;
    fwrite(&fileSize, 4, 1, outputFile);

    int32 reserved = 0x0000;
    fwrite(&reserved, 4, 1, outputFile);

    int32 dataOffset = HEADER_SIZE + INFO_HEADER_SIZE;
    fwrite(&dataOffset, 4, 1, outputFile);

    // Info Header
    int32 infoHeaderSize = INFO_HEADER_SIZE;
    fwrite(&infoHeaderSize, 4, 1, outputFile);
    fwrite(&width, 4, 1, outputFile);
    fwrite(&height, 4, 1, outputFile);
    int16 planes = 1;
    fwrite(&planes, 2, 1, outputFile);
    int16 bitsPerPixel = bytesPerPixel * 8;
    fwrite(&bitsPerPixel, 2, 1, outputFile);
    int32 compression = NO_COMPRESSION;
    fwrite(&compression, 4, 1, outputFile);
    int32 imageSize = width * height * bytesPerPixel;
    fwrite(&imageSize, 4, 1, outputFile);
    int32 resolutionX = 11811; // 300 dpi
    int32 resolutionY = 11811;
    fwrite(&resolutionX, 4, 1, outputFile);
    fwrite(&resolutionY, 4, 1, outputFile);
    int32 colorsUsed = MAX_NUMBER_OF_COLORS;
    fwrite(&colorsUsed, 4, 1, outputFile);
    int32 importantColors = ALL_COLORS_REQUIRED;
    fwrite(&importantColors, 4, 1, outputFile);

    // Guardar la imagen fila por fila (de abajo a arriba)
    int unpaddedRowSize = width * bytesPerPixel;
    for (int i = 0; i < height; i++) {
        int pixelOffset = ((height - i - 1) * unpaddedRowSize);
        fwrite(&pixels[pixelOffset], 1, unpaddedRowSize, outputFile);

        // Padding
        int padding = paddedRowSize - unpaddedRowSize;
        for (int j = 0; j < padding; j++) {
            fputc(0x00, outputFile);
        }
    }

    fclose(outputFile);
    return true;
}