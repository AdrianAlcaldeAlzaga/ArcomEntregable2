/*
* ARQUITECTURA DE COMPUTADORES
* Hecho por: Adrián Zamora Sánchez y Adrián Alcalde Alzaga
* Ejercicio: Entregable 2 de CUDA
* Descripción: Dibujar un tablero de ajedrez utilizando un kernel bidimensional de bloques de 16x16 hilos 
* y donde cada hilo se encargue de generar un pixel de la imagen final
*/

// includes
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <device_launch_parameters.h>
#include "gpu_bitmap.h"
// defines
#define NUMHILOS 16 //hilos del programa
#define ANCHO 512 // Dimension horizontal
#define ALTO 512 // Dimension vertical

void propiedades_Device(int deviceID)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);

	// calculo del numero de cores (SP)
	int cudaCores = 0;
	int SM = deviceProp.multiProcessorCount;
	int maxThreads = deviceProp.maxThreadsPerBlock;
	int major = deviceProp.major;
	int minor = deviceProp.minor;
	const char* archName;

	switch (major)
	{
	case 1:
		//TESLA
		archName = "TESLA";
		cudaCores = 8;
		break;
	case 2:
		//FERMI
		archName = "FERMI";
		if (minor == 0)
			cudaCores = 32;
		else
			cudaCores = 48;
		break;
	case 3:
		//KEPLER
		archName = "KEPLER";
		cudaCores = 192;
		break;
	case 5:
		//MAXWELL
		archName = "MAXWELL";
		cudaCores = 128;
		break;
	case 6:
		//PASCAL
		archName = "PASCAL";
		cudaCores = 64;
		break;
	case 7:
		//VOLTA(7.0) //TURING(7.5)
		cudaCores = 64;
		if (minor == 0)
			archName = "VOLTA";
		else
			archName = "TURING";
		break;
	case 8:
		// AMPERE
		archName = "AMPERE";
		cudaCores = 64;
		break;
	case 9:
		//HOPPER
		archName = "HOPPER";
		cudaCores = 64;
		break;
	default:
		//ARQUITECTURA DESCONOCIDA
		archName = "DESCONOCIDA";
	}

	int rtV;
	cudaRuntimeGetVersion(&rtV);

	// presentacion de propiedades
	printf("***************************************************\n");
	printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
	printf("***************************************************\n");
	printf("> CUDA Toolkit\t\t\t: %d.%d\n", rtV / 1000, (rtV % 1000) / 10);
	printf("> Arquitectura CUDA\t\t: %s\n", archName);
	printf("> Capacidad de Computo\t\t: %d.%d\n", major, minor);
	printf("> No. MultiProcesadores\t\t: %d\n", SM);
	printf("> No. Nucleos CUDA (%dx%d)\t: %d\n", cudaCores, SM, cudaCores * SM);
	printf("> Memoria Global (total)\t: %u MiB\n", deviceProp.totalGlobalMem / (1024 * 1024));
}


// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void kernel(unsigned char* imagen)
{
	// ** Kernel bidimensional multibloque **
	//
	// coordenada horizontal de cada hilo
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	// coordenada vertical de cada hilo
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// indice global de cada hilo (indice lineal para acceder a la memoria)
	int myID = x + y * blockDim.x * gridDim.x;
	// cada hilo obtiene la posicion de su pixel
	int miPixel = myID * 4;

	int tableroX = x / 64; // Cada cuadrado en el tablero tiene 64 píxeles en la dimensión x
	int tableroY = y / 64; // Cada cuadrado en el tablero tiene 64 píxeles en la dimensión y
	int tablero = (tableroX % 2) + (tableroY % 2); //posicion de cada pixel dependiendo de si es par o impar
	// cada hilo rellena los 4 canales de su pixel con un valor arbitrario
	if (tablero % 2 == 0)
	{
		imagen[miPixel + 0] = 0; // canal R
		imagen[miPixel + 1] = 0;// canal G
		imagen[miPixel + 2] = 0; // canal B
		imagen[miPixel + 3] = 0; // canal alfa
	}
	else
	{
		imagen[miPixel + 0] = 255; // canal R
		imagen[miPixel + 1] = 255;// canal G
		imagen[miPixel + 2] = 255; // canal B
		imagen[miPixel + 3] = 0; // canal alfa
	}
}
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{

	// Busqueda de dispositivos
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	// Muestra información de los dispositivos encontrados
	if (deviceCount == 0)
	{
		printf("!!!!!No se han encontrado dispositivos CUDA!!!!!\n");
		printf("<pulsa [INTRO] para finalizar>");
		getchar();
		return 1;
	}
	else
	{
		// Muestra los datos de cada dispositivo encontrado
		for (int id = 0; id < deviceCount; id++)
		{
			propiedades_Device(id);
		}
	}

	// Declaracion del bitmap:
	// Inicializacion de la estructura RenderGPU
	RenderGPU foto(ANCHO, ALTO);

	// Tamaño del bitmap en bytes
	size_t bmp_size = foto.image_size();

	// Asignacion y reserva de la memoria en el host (framebuffer)
	unsigned char* host_bitmap = foto.get_ptr();

	// Reserva en el device
	unsigned char* dev_bitmap;
	cudaMalloc((void**)&dev_bitmap, bmp_size);

	// Lanzamos un kernel bidimensional con bloques de 256 hilos (16x16)
	dim3 hilosB(NUMHILOS, NUMHILOS);

	// Calculamos el numero de bloques necesario (un hilo por cada pixel)
	dim3 Nbloques(ANCHO / NUMHILOS, ALTO / NUMHILOS);

	// Declaración del evento que calcula el tiempo de ejecución
	cudaEvent_t start;
	cudaEvent_t stop;

	// Creacion del evento
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Captura de la marca de tiempo de inicio
	cudaEventRecord(start, 0);

	// Generamos el bitmap
	kernel << <Nbloques, hilosB >> > (dev_bitmap);

	// Captura el final de la marca de tiempo
	cudaEventRecord(stop, 0);

	// Sincronizacion GPU-CPU
	cudaEventSynchronize(stop);

	// Calculo del tiempo en milisegundos
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	// Impresion de resultados
	printf("> Tiempo de ejecucion\t\t: %f ms\n", elapsedTime);
	printf("***************************************************\n");

	// Finalización del evento
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copiamos los datos desde la GPU hasta el framebuffer para visualizarlos
	cudaMemcpy(host_bitmap, dev_bitmap, bmp_size, cudaMemcpyDeviceToHost);

	// Visualizacion y salida
	// La funcion ″display_and_exit()″ no retorna e impide continuar con el main()
	foto.display_and_exit();

	return 0;
}
