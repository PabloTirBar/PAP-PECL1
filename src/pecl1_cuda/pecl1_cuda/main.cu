#include <iostream>
#include <string>
#include <cstdlib>
#include "utils.h"
#include "blanco_negro.cuh"
#include "pixelado.cuh"
#include "identificar_colores.cuh"
#include "filtrar_delinear.cuh"
#include "pseudo_hash.cuh"

using namespace std;

void mostrarMenu() {
    cout << "\n========= MENÚ DE OPCIONES =========" << endl;
    cout << "0. Salir" << endl;
    cout << "1. Crear copia de la imagen base" << endl;
    cout << "2. Conversión a blanco y negro" << endl;
    cout << "3. Pixelar imagen" << endl;
    cout << "4. Identificación de colores" << endl;
    cout << "5. Filtrado y delineado de zonas de color" << endl;
    cout << "6. Cálculo de pseudo-hash" << endl;
    cout << "====================================" << endl;
    cout << "Seleccione una opción: ";
}

int main() {
    string rutaImagen;
    cout << "Introduce la ruta de la imagen BMP (ENTER para usar la ruta por defecto): ";
    getline(cin, rutaImagen);
    if (rutaImagen.empty()) {
        rutaImagen = "C:\\Users\\iubal.camjalli\\Documents\\PAP\\PAP-PECL1\\imgs\\input.bmp";
    }

    // Variables comunes para todas las fases
    byte* pixels;
    int32 ancho, alto, bytesPerPixel;

    if (!cargarBMP(rutaImagen.c_str(), &pixels, &ancho, &alto, &bytesPerPixel)) {
        cerr << "No se pudo cargar la imagen base. Terminando programa." << endl;
        return -1;
    }

    int opcion;
    do {
        mostrarMenu();
        cin >> opcion;
        cin.ignore(); // Limpiar el buffer

        switch (opcion) {
        case 1: {
            string rutaSalida;
            cout << "Ruta para guardar la copia: ";
            getline(cin, rutaSalida);
            if (!exportarBMP(rutaSalida.c_str(), pixels, ancho, alto, bytesPerPixel)) {
                cerr << "No se pudo guardar la copia de la imagen." << endl;
            }
            else {
                cout << "Imagen copiada con éxito en: " << rutaSalida << endl;
            }
            break;
        }
        case 2: {
            convertirBlancoNegro(pixels, ancho, alto, bytesPerPixel);
            string rutaSalida = "C:\\Users\\iubal.camjalli\\Documents\\PAP\\PAP-PECL1\\imgs\\blanco_y_negro.bmp";
            if (!exportarBMP(rutaSalida.c_str(), pixels, ancho, alto, bytesPerPixel)) {
                cerr << "Error al guardar la imagen blanco y negro." << endl;
            }
            else {
                cout << "Imagen en blanco y negro guardada en: " << rutaSalida << endl;
            }
            break;
        }
        case 3: {
            pixelarImagen(pixels, ancho, alto, bytesPerPixel, 8); // Tamaño de bloque 8 como ejemplo
            string rutaSalida = "C:\\Users\\iubal.camjalli\\Documents\\PAP\\PAP-PECL1\\imgs\\pixelado.bmp";
            if (!exportarBMP(rutaSalida.c_str(), pixels, ancho, alto, bytesPerPixel)) {
                cerr << "Error al guardar la imagen pixelada." << endl;
            }
            else {
                cout << "Imagen pixelada guardada en: " << rutaSalida << endl;
            }
            break;
        }
        case 4: {
            float umbral, magnitud;

            cout << "Introduce el umbral (valor por defecto 30): ";
            string input;
            getline(cin, input);
            umbral = input.empty() ? 30.0f : std::stof(input);

            cout << "Introduce el factor de magnitud (valor por defecto 1.0): ";
            getline(cin, input);
            magnitud = input.empty() ? 1.0f : std::stof(input);

            byte* copiaRojo = (byte*)malloc(ancho * alto * bytesPerPixel);
            byte* copiaVerde = (byte*)malloc(ancho * alto * bytesPerPixel);
            byte* copiaAzul = (byte*)malloc(ancho * alto * bytesPerPixel);
            memcpy(copiaRojo, pixels, ancho * alto * bytesPerPixel);
            memcpy(copiaVerde, pixels, ancho * alto * bytesPerPixel);
            memcpy(copiaAzul, pixels, ancho * alto * bytesPerPixel);

            identificarColor(copiaRojo, ancho, alto, bytesPerPixel, ROJO, umbral, magnitud,
                "C:\\Users\\iubal.camjalli\\Documents\\PAP\\PAP-PECL1\\imgs\\identificar_rojo.bmp");

            identificarColor(copiaVerde, ancho, alto, bytesPerPixel, VERDE, umbral, magnitud,
                "C:\\Users\\iubal.camjalli\\Documents\\PAP\\PAP-PECL1\\imgs\\identificar_verde.bmp");

            identificarColor(copiaAzul, ancho, alto, bytesPerPixel, AZUL, umbral, magnitud,
                "C:\\Users\\iubal.camjalli\\Documents\\PAP\\PAP-PECL1\\imgs\\identificar_azul.bmp");

            free(copiaRojo);
            free(copiaVerde);
            free(copiaAzul);
            break;
        }
        case 5: {
            float umbral, magnitud;

            cout << "Introduce el umbral (valor por defecto 30): ";
            string input;
            getline(cin, input);
            umbral = input.empty() ? 30.0f : std::stof(input);

            cout << "Introduce el factor de magnitud (valor por defecto 1.0): ";
            getline(cin, input);
            magnitud = input.empty() ? 1.0f : std::stof(input);

            byte* copiaRojo = (byte*)malloc(ancho * alto * bytesPerPixel);
            byte* copiaVerde = (byte*)malloc(ancho * alto * bytesPerPixel);
            byte* copiaAzul = (byte*)malloc(ancho * alto * bytesPerPixel);
            memcpy(copiaRojo, pixels, ancho * alto * bytesPerPixel);
            memcpy(copiaVerde, pixels, ancho * alto * bytesPerPixel);
            memcpy(copiaAzul, pixels, ancho * alto * bytesPerPixel);

            filtrarYDelimitarColor(copiaRojo, ancho, alto, bytesPerPixel, "rojo", umbral, magnitud);
            filtrarYDelimitarColor(copiaVerde, ancho, alto, bytesPerPixel, "verde", umbral, magnitud);
            filtrarYDelimitarColor(copiaAzul, ancho, alto, bytesPerPixel, "azul", umbral, magnitud);

            free(copiaRojo);
            free(copiaVerde);
            free(copiaAzul);
            break;
        }
        case 6: {
            mostrarPseudoHash(pixels, ancho, alto, bytesPerPixel);
            break;
        }
        case 0:
            cout << "Saliendo del programa..." << endl;
            break;
        default:
            cout << "Opción no válida. Intente de nuevo." << endl;
        }

    } while (opcion != 0);

    free(pixels);
    return 0;
}