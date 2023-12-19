#include <iostream>
#include <stdio.h>

__global__ void tmp_kernel() {
    printf("hello\n");
}

int main() {
    tmp_kernel<<<1, 1>>>();
    return 0;
}
