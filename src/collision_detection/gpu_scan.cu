#include <iostream>

#include "collision_detection/gpu_scan.cuh"
#include "numeric"

/*
 * A simple implementation of GPU Scan Algorithm proposed in paper
 *      Single-pass Parallel Prefix Scan with Decoupled Look-back
 *
 *      Step1: Initialize Partition Descriptors
 *
 *      Step2: For each tile,
 *          (a) reduce and got partial
 *          (b) look back to get inclusive sum
 *          (c) perform scan with inclusive and sum per thread in (a).
 */

/*
 * Step1: Initialize Tile Descriptors
 */

enum ScanTileStatus {
  SCAN_TILE_INVALID,
  SCAN_TILE_PARTIAL,
  SCAN_TILE_INCLUSIVE
};

constexpr const int MAX_TILES = 20000;

__device__ ScanTileStatus tile_status[MAX_TILES];
__device__ int tile_partial[MAX_TILES];
__device__ int tile_inclusive[MAX_TILES];

constexpr const int THREAD_PER_BLOCK = 32;
constexpr const int ITEMS_PER_THREAD = 128;

__device__ void initialize_status(int num_tiles) {
  int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tile_idx < num_tiles) {
    tile_status[tile_idx] = ScanTileStatus::SCAN_TILE_INVALID;
  }
}

/*
 * Step2:
 */

__device__ void consume_tile(const int *input, int *output) {
  int items[ITEMS_PER_THREAD];
  __shared__ int block_sum[THREAD_PER_BLOCK];

  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    items[i] = input[global_tid * ITEMS_PER_THREAD + i];
  }

  for (int i = 1; i < ITEMS_PER_THREAD; i++) {
    items[i] += items[i - 1];
  }

  block_sum[threadIdx.x] = items[ITEMS_PER_THREAD - 1];

  __threadfence_block();

  if (threadIdx.x == 0) {
    int partial = 0;

    for (int i : block_sum) {
      partial += i;
    }

    // @todo: GPU reduction
    for (int i = 1; i < THREAD_PER_BLOCK; i++) {
      block_sum[i] += block_sum[i - 1];
    }

    tile_partial[blockIdx.x] = partial;
    //        printf("initialized %d %d %d %d\n", blockIdx.x, partial,
    //        input[global_tid * ITEMS_PER_THREAD], blockDim.x);
    tile_status[blockIdx.x] = ScanTileStatus::SCAN_TILE_PARTIAL;
    __threadfence();
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) {
      tile_inclusive[0] = tile_partial[0];
      tile_status[0] = ScanTileStatus::SCAN_TILE_INCLUSIVE;
      __threadfence();
    } else {
      int j = blockIdx.x - 1;
      int exclusive = 0;
      while (j >= 0 and tile_status[j] != ScanTileStatus::SCAN_TILE_INCLUSIVE) {
        auto status = tile_status[j];
        do {
          __threadfence();
          status = tile_status[j];
        } while (j != 0 and status == ScanTileStatus::SCAN_TILE_INVALID);

        if (j == 0 || tile_status[j] == ScanTileStatus::SCAN_TILE_PARTIAL) {
          exclusive += tile_partial[j];
          //                    printf("add %d %d %d\n",blockIdx.x , j,
          //                    tile_partial[j]);
          __threadfence();
          j--;
        } else {
          break;
        }
      }
      if (j >= 0) {
        exclusive += tile_inclusive[j];
        //                printf("rel %d %d %d %d %d\n", blockIdx.x, j,
        //                exclusive, tile_inclusive[j], tile_status[j]);
      }
      tile_inclusive[blockIdx.x] = exclusive + tile_partial[blockIdx.x];
      tile_status[blockIdx.x] = ScanTileStatus::SCAN_TILE_INCLUSIVE;
      __threadfence();
    }
  }

  __syncthreads();

  int exclusive = tile_inclusive[blockIdx.x] - tile_partial[blockIdx.x];

  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    if (threadIdx.x > 0)
      output[global_tid * ITEMS_PER_THREAD + i] =
          items[i] + exclusive + block_sum[threadIdx.x - 1];
    else
      output[global_tid * ITEMS_PER_THREAD + i] = items[i] + exclusive;
  }
}

__global__ void scan_init(int n) { initialize_status(n); }

__global__ void scan(int *input, int *output) { consume_tile(input, output); }

void gpu_scan(int *input, int *output, int n) {
  int block_num = n / (ITEMS_PER_THREAD * THREAD_PER_BLOCK);
  scan_init<<<block_num, 1>>>(block_num);
  scan<<<block_num, THREAD_PER_BLOCK>>>(input, output);
}