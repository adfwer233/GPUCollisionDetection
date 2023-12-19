#include <iostream>
#include <stdio.h>

#include "collision_detection/gpu_sort.h"

__global__ void tmp_kernel() {
    printf("hello from kernel\n");
}

void tmp_function() {
    tmp_kernel<<<1, 1>>>();
}