using CUDA
using BenchmarkTools
using Test

N = 2^20
x_d = CUDA.fill(1.0f0, N)   # a vector stored on the GPU filled with 1.0 (Float32)

# CuDeviceArray is meant to be modified from the host side
function reverse_kernel!(a::CuDeviceArray{T}) where T
    i = threadIdx().x
    b = CuStaticSharedArray(T, 2)
    @inbounds b[2-i+1] = a[i] # bounds checking
    sync_threads()
    @inbounds a[i] = b[i]
    sync_warp()
    return
end

@cuda threads=2 reverse_kernel!(a)

# examine device code
@device_code dir="./device_code" @cuda threads=2 reverse_kernel!(a)