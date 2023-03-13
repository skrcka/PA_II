#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 10000 // Number of individuals
#define n 3    // Dimensionality of world

typedef struct {
    float* position;
    float* velocity;
    int* target_individuals;
    int common_target_individual;
} World;

__global__ void simulate_step(World* world, float* random_vector) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M) {
        // Step 1: Adjust velocities by random vector
        for (int i = 0; i < n; i++) {
            world->velocity[tid * n + i] += random_vector[i];
        }

        // Step 2: Move 5% towards pre-selected individuals
        int target = world->target_individuals[tid];
        for (int i = 0; i < n; i++) {
            world->position[tid * n + i] += 0.05 * (world->position[target * n + i] - world->position[tid * n + i]);
        }

        // Step 3: Move 10% towards common pre-selected individual
        int common_target = world->common_target_individual;
        for (int i = 0; i < n; i++) {
            world->position[tid * n + i] += 0.1 * (world->position[common_target * n + i] - world->position[tid * n + i]);
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
    h_world.common_target_individual = rand() % M;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < n; j++) {
            h_world.position[i * n + j] = (float)rand() / RAND_MAX;
            h_world.velocity[i * n + j] = 0.0;
        }
        h_world.target_individuals[i] = rand() % M;
    }

    // Allocate memory on GPU
    World* d_world;
    cudaMalloc((void**)&d_world, sizeof(World));
    cudaMalloc((void**)&d_world->position, n * M * sizeof(float));
    cudaMalloc((void**)&d_world->velocity, n * M * sizeof(float));
    cudaMallocManaged(&d_world->target_individual, M * sizeof(int));
    cudaMemcpy(d_world, &h_world, sizeof(World), cudaMemcpyHostToDevice);
    cudaMemcpy(d_world->position, h_world.position, n * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_world->velocity, h_world.velocity, n * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_world->target_individuals, h_world.target_individuals, M * sizeof(int), cudaMemcpyHostToDevice);

    // Run simulation
    int num_blocks = (M + 255) / 256;
    int num_threads_per_block = 256;
    for (int step = 0; step < 100; step++) {
        simulate_step<<<num_blocks, num_threads_per_block>>>(d_world, d_random_vector);
    }
    cudaDeviceSynchronize();

    // Copy data back to host
    cudaMemcpy(h_world.position, d_world->position, n * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_world.velocity, d_world->velocity, n * M * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(d_world->position);
    cudaFree(d_world->velocity);
    cudaFree(d_world->target_individuals);
    cudaFree(d_world);

    // Free memory on host
    free(h_world.position);
    free(h_world.velocity);
    free(h_world.target_individuals);

    return 0;
}
