#include "gtest/gtest.h"

#include "collision_detection/gpu_sort.cuh"

#include "thrust/device_vector.h"
#include "cub/cub.cuh"

TEST(HelloTest, BasicAssertions) {
    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
}

TEST(SortTest, TmpTest) {
    tmp_function();
    EXPECT_TRUE(true);
}

struct custom_t
{
    float f;
    int unused;
    long long int lli;

    custom_t() = default;
    custom_t(float f, long long int lli)
            : f(f)
            , unused(42)
            , lli(lli)
    {}
};

struct decomposer_t
{
    __host__ __device__ ::cuda::std::tuple<float &, long long int &>
    operator()(custom_t &key) const
    {
        return {key.f, key.lli};
    }
};

TEST(SortTest, CustomTypeRadixSort) {


    constexpr int num_items = 6;

    thrust::device_vector<custom_t> in = {
            {+2.5f, 4},
            {-2.5f, 0},
            {+1.1f, 3},
            {+0.0f, 1},
            {-0.0f, 2},
            {+3.7f, 5}
    };

    thrust::device_vector<custom_t> out(num_items);

    const custom_t *d_in = thrust::raw_pointer_cast(in.data());
    custom_t *d_out      = thrust::raw_pointer_cast(out.data());

// 1) Get temp storage size
    std::uint8_t *d_temp_storage{};
    std::size_t temp_storage_bytes{};

    cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                   temp_storage_bytes,
                                   d_in,
                                   d_out,
                                   num_items,
                                   decomposer_t{});

// 2) Allocate temp storage
    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

// 3) Sort keys
    cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                   temp_storage_bytes,
                                   d_in,
                                   d_out,
                                   num_items,
                                   decomposer_t{});

    thrust::device_vector<custom_t> expected_output = {
            {-2.5f, 0},
            {+0.0f, 1},
            {-0.0f, 2},
            {+1.1f, 3},
            {+2.5f, 4},
            {+3.7f, 5}
    };
}