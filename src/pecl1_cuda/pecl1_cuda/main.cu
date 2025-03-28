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
    cout << "\n========= MEN� DE OPCIONES =========" << endl;
    cout << "0. Salir" << endl;
    cout << "1. Crear copia de la imagen base" << endl;
    cout << "2. Conversi�n a blanco y negro" << endl;
    cout << "3. Pixelar imagen" << endl;
    cout << "4. Identificaci�n de colores" << endl;
    cout << "5. Filtrado y delineado de zonas de color" << endl;
    cout << "6. C�lculo de pseudo-hash" << endl;
    cout << "====================================" << endl;
    cout << "Seleccione una opci�n: ";
}

std::string normalizarSeparadores(const std::string& ruta) {
    std::string normalizada = ruta;
    for (char& c : normalizada) {
        if (c == '\\') c = '/';
    }
    return normalizada;
}

std::string obtenerNombreArchivo(const std::string& rutaCompleta) {
    std::string ruta = normalizarSeparadores(rutaCompleta);
    size_t pos = ruta.rfind('/');
    std::string nombre = (pos == std::string::npos) ? ruta : ruta.substr(pos + 1);

    size_t punto = nombre.rfind('.');
    if (punto != std::string::npos) {
        nombre = nombre.substr(0, punto);
    }
    return nombre;
}

int main() {
    string rutaImagen;
    cout << "Introduce la ruta de la imagen BMP (ENTER para usar la ruta por defecto): ";
    getline(cin, rutaImagen);
    if (rutaImagen.empty()) {
        rutaImagen = "..\\imgs\\input.bmp";
    }

    string nombreArchivo = obtenerNombreArchivo(rutaImagen);
    cout << nombreArchivo << endl;
    // Variables comunes para todas las fases
    byte* pixels;
    int32 ancho, alto, bytesPerPixel;



    int opcion;
    do {
        if (!cargarBMP(rutaImagen.c_str(), &pixels, &ancho, &alto, &bytesPerPixel)) {
            cerr << "No se pudo cargar la imagen base. Terminando programa." << endl;
            return -1;
        }
        mostrarMenu();
        cin >> opcion;
        cin.ignore();

        switch (opcion) {
        case 1: {
            string rutaSalida = "..\\imgs\\" + nombreArchivo + "_copia.bmp";
            if (!exportarBMP(rutaSalida.c_str(), pixels, ancho, alto, bytesPerPixel)) {
                cerr << "No se pudo guardar la copia de la imagen." << endl;
            }
            else {
                cout << "Imagen copiada con �xito en: " << rutaSalida << endl;
            }
            break;
        }
        case 2: {
            convertirBlancoNegro(pixels, ancho, alto, bytesPerPixel);
            string rutaSalida = "..\\imgs\\" + nombreArchivo + "_blanco_y_negro.bmp";
            if (!exportarBMP(rutaSalida.c_str(), pixels, ancho, alto, bytesPerPixel)) {
                cerr << "Error al guardar la imagen blanco y negro." << endl;
            }
            else {
                cout << "Imagen en blanco y negro guardada en: " << rutaSalida << endl;
            }
            break;
        }
        case 3: {
            int tamanoPixel;
            cout << "Introduce el tama�o del p�xel para el pixelado (por defecto 8): ";
            string input;
            getline(cin, input);
            tamanoPixel = input.empty() ? 8 : stoi(input);

            pixelarImagen(pixels, ancho, alto, bytesPerPixel, tamanoPixel);
            string rutaSalida = "..\\imgs\\" + nombreArchivo + "_pixelado.bmp";
            if (!exportarBMP(rutaSalida.c_str(), pixels, ancho, alto, bytesPerPixel)) {
                cerr << "Error al guardar la imagen pixelada." << endl;
            }
            else {
                cout << "Imagen pixelada guardada en: " << rutaSalida << endl;
            }
            break;
        }
        case 4: {
            float umbral;
            cout << "Introduce el umbral (valor por defecto 30): ";
            string input;
            getline(cin, input);
            umbral = input.empty() ? 30.0f : std::stof(input);



            byte* copiaRojo = (byte*)malloc(ancho * alto * bytesPerPixel);
            byte* copiaVerde = (byte*)malloc(ancho * alto * bytesPerPixel);
            byte* copiaAzul = (byte*)malloc(ancho * alto * bytesPerPixel);
            memcpy(copiaRojo, pixels, ancho * alto * bytesPerPixel);
            memcpy(copiaVerde, pixels, ancho * alto * bytesPerPixel);
            memcpy(copiaAzul, pixels, ancho * alto * bytesPerPixel);

            identificarColor(copiaRojo, ancho, alto, bytesPerPixel, ROJO, umbral,
                ("..\\imgs\\" + nombreArchivo + "_identificar_rojo.bmp").c_str());

            identificarColor(copiaVerde, ancho, alto, bytesPerPixel, VERDE, umbral,
                ("..\\imgs\\" + nombreArchivo + "_identificar_verde.bmp").c_str());

            identificarColor(copiaAzul, ancho, alto, bytesPerPixel, AZUL, umbral,
                ("..\\imgs\\" + nombreArchivo + "_identificar_azul.bmp").c_str());

            free(copiaRojo);
            free(copiaVerde);
            free(copiaAzul);
            break;
        }
              case 5: {
                    float umbral;
                    int halo;
                    std::string input, colorStr;
                    ColorDetectado color;

                    cout << "Introduce el color a filtrar (rojo / verde / azul): ";
                    getline(cin, colorStr);

                    if (colorStr == "rojo") color = ROJO;
                    else if (colorStr == "verde") color = VERDE;
                    else if (colorStr == "azul") color = AZUL;
                    else {
                        cerr << "Color no reconocido. Debe ser: rojo, verde o azul." << endl;
                        break;
                    }

                    cout << "Introduce el umbral (por defecto 30): ";
                    getline(cin, input);
                    umbral = input.empty() ? 30.0f : std::stof(input);

                    cout << "Introduce el tama�o del halo (por defecto 15): ";
                    getline(cin, input);
                    halo = input.empty() ? 15 : std::stoi(input);

                    byte* copia = (byte*)malloc(ancho * alto * bytesPerPixel);
                    memcpy(copia, pixels, ancho * alto * bytesPerPixel);

                    string ruta = "..\\imgs\\" + nombreArchivo;
                    if (color == ROJO) ruta += "_delimitar_identificar_rojo.bmp";
                    else if (color == VERDE) ruta += "_delimitar_identificar_verde.bmp";
                    else ruta += "_delimitar_identificar_azul.bmp";

                    filtrarYDelimitarColor(copia, ancho, alto, bytesPerPixel, color, umbral, halo, ruta.c_str());

                    free(copia);
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
            cout << "Opci�n no v�lida. Intente de nuevo." << endl;
        }

    } while (opcion != 0);

    free(pixels);
    return 0;
}