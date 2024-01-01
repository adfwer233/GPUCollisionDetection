#include "gtest/gtest.h"

#include "numeric"
#include "iterator"
#include "chrono"
#include "format"
#include "collision_detection/gpu_scan.cuh"

#include "cub/cub.cuh"

TEST(ScanTest, CCCLTest) {

    using namespace std::chrono;

    std::vector<int> data;

    const int thread_per_block = 32;
    const int block_num = 16384;
    const int item_per_thread = 32;

    constexpr int n = thread_per_block * block_num * item_per_thread;
    std::cout << n << std::endl;
    for (int i = 0; i < n; i++) data.emplace_back(1);

    std::vector<int> ans;

    auto cpu_start = system_clock::now();
    std::inclusive_scan(data.begin(), data.end(), std::back_inserter(ans));
    auto cpu_end = system_clock::now();
    auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start);
    auto cpu_time = double(cpu_duration.count()) * microseconds::period::num / microseconds::period::den;

    int *dev_input, *output;

    output = (int*)malloc(n * sizeof(int));

    cudaMalloc(&dev_input, n * sizeof(int));

    cudaMemcpy(dev_input, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    auto gpu_start = system_clock::now();

    void *device_temp_storage = nullptr;
    size_t temp_storage_bytes{0};
    cub::DeviceScan::InclusiveSum(device_temp_storage, temp_storage_bytes, dev_input, n);
    cudaMalloc(&device_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(device_temp_storage, temp_storage_bytes, dev_input, n);

    auto gpu_end = system_clock::now();

    cudaMemcpy(output, dev_input,  n * sizeof(int), cudaMemcpyDeviceToHost);

    auto gpu_duration = duration_cast<microseconds>(gpu_end - gpu_start);
    auto gpu_time = double(gpu_duration.count()) * microseconds::period::num / microseconds::period::den;

    std::cout << std::format("time: cpu {}ms, gpu {}ms", cpu_time, gpu_time) << std::endl;

    std::cout << std::endl;

    std::vector<int> cuda_res;
    std::copy(output, output + n, std::back_inserter(cuda_res));

    for (int i = 0 ; i < data.size(); i++) {
        if (ans[i] != cuda_res[i]) {
            std::cout << std::format("idx {}, ans {}, cuda {}", i, ans[i], cuda_res[i]) << std::endl;
            break;
        }
    }

    EXPECT_TRUE(cuda_res == ans);

}

TEST(ScanTest, SimpleTest) {
    using namespace std::chrono;

    std::vector<int> data;

    const int thread_per_block = 32;
    const int block_num = 16384;
    const int item_per_thread = 32;

    constexpr int n = thread_per_block * block_num * item_per_thread;
    std::cout << n << std::endl;
    for (int i = 0; i < n; i++) data.emplace_back(1);

    std::vector<int> ans;

    auto cpu_start = system_clock::now();
    std::inclusive_scan(data.begin(), data.end(), std::back_inserter(ans));
    auto cpu_end = system_clock::now();
    auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start);
    auto cpu_time = double(cpu_duration.count()) * microseconds::period::num / microseconds::period::den;

    int *dev_input, *dev_output, *output;

    output = (int*)malloc(n * sizeof(int));

    cudaMalloc(&dev_input, n * sizeof(int));
    cudaMalloc(&dev_output, n * sizeof(int));


    cudaMemcpy(dev_input, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    auto gpu_start = system_clock::now();
    gpu_scan(dev_input, dev_output, n);
    auto gpu_end = system_clock::now();
    cudaMemcpy(output, dev_output,  n * sizeof(int), cudaMemcpyDeviceToHost);

    auto gpu_duration = duration_cast<microseconds>(gpu_end - gpu_start);
    auto gpu_time = double(gpu_duration.count()) * microseconds::period::num / microseconds::period::den;

    std::cout << std::format("time: cpu {}ms, gpu {}ms", cpu_time, gpu_time) << std::endl;

    std::cout << std::endl;

    std::vector<int> cuda_res;
    std::copy(output, output + n, std::back_inserter(cuda_res));

//    std::ranges::copy(cuda_res, std::ostream_iterator<int>(std::cout, " "));
//    std::cout << std::endl;
//    std::ranges::copy(ans, std::ostream_iterator<int>(std::cout, " "));
//    std::cout << std::endl;
//    std::ranges::copy(data, std::ostream_iterator<int>(std::cout, " "));

    for (int i = 0 ; i < data.size(); i++) {
        if (ans[i] != cuda_res[i]) {
            std::cout << std::format("idx {}, ans {}, cuda {}", i, ans[i], cuda_res[i]) << std::endl;
            break;
        }
    }

    EXPECT_TRUE(cuda_res == ans);
}