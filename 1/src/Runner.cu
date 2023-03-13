#include <cudaDefs.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <benchmark.h>

using gpubenchmark::print_time;

#define COMPUTE_SAFE(x) x

#define M 1000 // Number of individuals
#define n 100    // Dimensionality of world
#define STEP_COUNT 1

typedef struct {
    float* position;
    float* velocity;
    int* target_individuals;
} World;

__device__ World d_world;

__global__ void simulate_step(float* random_vector, int common_target_individual) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M) {
        // Step 1: Adjust velocities by random vector
        for (int i = 0; i < n; i++) {
            d_world.velocity[tid * n + i] += random_vector[i];
        }

        // Step 2: Move points
        for (int i = 0; i < n; i++) {
            d_world.position[tid * n + i] += d_world.velocity[tid * n + i];
        }

        // Step 3: Move 5% towards pre-selected individuals
        int target = d_world.target_individuals[tid];
        for (int i = 0; i < n; i++) {
            d_world.position[tid * n + i] += 0.05 * (d_world.position[target * n + i] - d_world.position[tid * n + i]);
        }
        
        // Step 4: Move 10% towards common pre-selected individual
        for (int i = 0; i < n; i++) {
            d_world.position[tid * n + i] += 0.1 * (d_world.position[common_target_individual * n + i] - d_world.position[tid * n + i]);
        }
    }
}

int main() {
    // Initialize random vector
    float h_random_vector[n];
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        h_random_vector[i] = (float)rand() / RAND_MAX;
    }
    float* d_random_vector;
    cudaMalloc((void**)&d_random_vector, n * sizeof(float));
    cudaMemcpy(d_random_vector, h_random_vector, n * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize world
    World h_world;
    h_world.position = (float*)malloc(n * M * sizeof(float));
    h_world.velocity = (float*)malloc(n * M * sizeof(float));
    h_world.target_individuals = (int*)malloc(M * sizeof(int));
    int common_target_individual = rand() % M;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < n; j++) {
            h_world.position[i * n + j] = (float)rand() / RAND_MAX;
            h_world.velocity[i * n + j] = (float)rand() / RAND_MAX;
        }
        h_world.target_individuals[i] = rand() % M;
    }

    /*
    for (int j = 0; j < n; j++)
    {
        printf("%f ", h_random_vector[j]);
    }
    printf("\n");
    printf("\n");

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%f ", h_world.position[i * n + j]);
            printf("%f ", h_world.velocity[i * n + j]);
        }
        printf("\n");
    }

    printf("\n");
    */

    // Allocate memory on GPU
    World c_world;
    //checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(World)));
    checkCudaErrors(cudaMalloc(&(c_world.position), n * M * sizeof(float)));
    checkCudaErrors(cudaMalloc(&(c_world.velocity), n * M * sizeof(float)));
    checkCudaErrors(cudaMalloc(&(c_world.target_individuals), M * sizeof(int)));
    
    checkCudaErrors(cudaMemcpy(c_world.position, h_world.position, n * M * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(c_world.velocity, h_world.velocity, n * M * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(c_world.target_individuals, h_world.target_individuals, M * sizeof(int), cudaMemcpyHostToDevice));
    COMPUTE_SAFE(cudaMemcpyToSymbol(d_world, &c_world, sizeof(World)));
    //checkCudaErrors(cudaMemcpy(&d_world, &h_world, sizeof(World), cudaMemcpyHostToDevice));

    // Run simulation
    int num_blocks = (M + 255) / 256;
    int num_threads_per_block = 256;

    auto test1 = [&]()
    {
        simulate_step << <num_blocks, num_threads_per_block >> > (d_random_vector, common_target_individual);
    };

    print_time("simulate_step", test1, 100);
    /*
    for (int step = 0; step < STEP_COUNT; step++) {
        simulate_step << <num_blocks, num_threads_per_block >> > (d_random_vector, common_target_individual);
    }
    */
    cudaDeviceSynchronize();

    // Copy data back to host
    //checkCudaErrors(cudaMemcpy(&h_world, &c_world, sizeof(World), cudaMemcpyDeviceToHost));
    //COMPUTE_SAFE(cudaMemcpyToSymbol(c_world, &d_world, sizeof(World)));
    checkCudaErrors(cudaMemcpy(h_world.position, c_world.position, n * M * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_world.velocity, c_world.velocity, n * M * sizeof(float), cudaMemcpyDeviceToHost));

    /*
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%f ", h_world.position[i * n + j]);
            printf("%f ", h_world.velocity[i * n + j]);
        }
        printf("\n");
    }
    */

    // Free memory on GPU
    cudaFree(d_world.position);
    cudaFree(d_world.velocity);
    cudaFree(d_world.target_individuals);
    //cudaFree(d_world);

    // Free memory on host
    free(h_world.position);
    free(h_world.velocity);
    free(h_world.target_individuals);

    return 0;
}
