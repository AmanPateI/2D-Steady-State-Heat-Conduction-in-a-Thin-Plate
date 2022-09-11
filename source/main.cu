/*
Author: Aman Patel
Last Date Modified: 11/21/2021
Description: Calculates the heat distribution on a thin plate using cuda events.
*/
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <getopt.h>
#include <math.h>
#include <fstream>


/*
* calculates the heat conduction in a thin metal plate 
* implementing the code provided in the lab document 
*/
__global__ void heatConduction(const double *__restrict__ G, double *__restrict__ H, const int interiorX, const int interiorY)
{
    // iterating the function given in lab5 pdf. 
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < interiorX - 1 && j > 0 && j < interiorY - 1)
        H[i + j * interiorX] = 0.25 * (G[(i + 1) + j * interiorX] + G[(i - 1) + j * interiorX] + G[i + (j + 1) * interiorX] + G[i + (j - 1) * interiorX]);
}

/*
* Initilize grid
* initial interior temperature of 20°C 
* fixed boundary temperature of 100* celcious 
*/
void Initialize(double *__restrict h_Matrix, const int interiorX, const int interiorY)
{

    for (int j = 0; j < interiorY; j++)
        for (int i = 0; i < interiorX; i++)
            h_Matrix[i + interiorX * j] = 20.0;

    for (int i = 0; i < interiorX; i++)
        if (i > (int)(0.3 * interiorX) && i < (int)(0.7 * interiorX))
            h_Matrix[i] = 100.0;
}

/*
* main function
* simulates the heat transfer onto thin plate
* Takes in parameters N and I
*/ 
int main(int argc, char *argv[])
{
    int option;
    int N = 0;
    int I = 0;
    // -N = interior dimension -I = total iterations
    while ((option = getopt(argc, argv, "N:I:")) != -1)
    {
        switch (option)
        {
        case 'N':
            N = atoi(optarg);
            break;
        case 'I':
            I = atoi(optarg);
            break;
        default:
            std::cout << "Invalid parameters, please check your values." << std::endl;
            return 1;
        }
    }
    // Check for invalid input
    if (N == 0 || I == 0)
    {
        std::cout << "Invalid parameters, please check your values." << std::endl;
        return 1;
    }
    if (N < 0 || I < 0)
    {
        std::cout << "Invalid parameters, please check your values." << std::endl;
        return 1;
    }

    // constants to assit with matrix shifting
    const int interiorX = N + 2;
    const int interiorY = interiorX;
    int matrixShift = 2;
    // given from input parameter -I
    const int totalIterations = I;
  
    // matrix reprsenting the heated points of the plate
    double *heatMatrix = (double *)calloc(interiorX * interiorY, sizeof(double));
    Initialize(heatMatrix, interiorX, interiorY);
    double *h_Matrix_GPU = (double *)malloc(interiorX * interiorY * sizeof(double));

    double *cudaMatrix;
    // creating a matrix with cuda on the GPU
    cudaMalloc((void **)&cudaMatrix, interiorX * interiorY * sizeof(double));
    double *cudaMatrix_X;
    cudaMalloc((void **)&cudaMatrix_X, interiorX * interiorY * sizeof(double));
    // set both arrays equal to each other using cuda command Memcpy
    cudaMemcpy(cudaMatrix, heatMatrix, interiorX * interiorY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaMatrix_X, cudaMatrix, interiorX * interiorY * sizeof(double), cudaMemcpyDeviceToDevice);

    // creates a cuda dimBLock of size 16*16
    dim3 dimBlock(16, 16);
    dim3 dimGrid(ceil((interiorX + 2) / (16)) + 1, ceil((interiorY + 2) / (16) + 1));

    // timer to be outputed to terminal
    float time;
    // begin running the cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // events are ran the number of total iterations -I
    for (int i = 0; i < totalIterations; i += matrixShift)
    {
        heatConduction<<<dimGrid, dimBlock>>>(cudaMatrix, cudaMatrix_X, interiorX, interiorY); // calculating the first part of the alogrithem 
        heatConduction<<<dimGrid, dimBlock>>>(cudaMatrix_X,cudaMatrix, interiorX, interiorY); // transfering to the seocnd part 
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout.precision(3);
    // output the time to the console
    std::cout << "Time: " << time << "ms" << std::fixed << std::endl;

    cudaMemcpy(h_Matrix_GPU, cudaMatrix, interiorX * interiorY * sizeof(double), cudaMemcpyDeviceToHost);

    // out stream the results to a csv file
    std::ofstream myfile;
    myfile.open("finalTemperatures.csv");
    std::cout.precision(6);
    for (int j = 0; j < interiorY; j++)
    {
        for (int i = 0; i < interiorX; i++)
        {
            myfile << h_Matrix_GPU[j * interiorX + i] << "," << std::fixed;
        }
        myfile << std::endl;
    }
    myfile.close();
    
    //free up taken memory

    free(heatMatrix);
    free(h_Matrix_GPU);

    cudaFree(cudaMatrix);
    cudaFree(cudaMatrix_X);

    return 0;
}