#include <stdio.h>

#include <iostream>

#include "collision_detection/gpu_sort.cuh"

__global__ void tmp_kernel() { printf("hello from kernel\n"); }

void accumulate(std::vector<float> &array) {}

void tmp_function() { tmp_kernel<<<1, 1>>>(); }