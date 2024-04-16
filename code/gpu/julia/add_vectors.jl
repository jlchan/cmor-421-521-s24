using CUDA
using BenchmarkTools
using Test

N = 2^20
x_d = CUDA.fill(1.0f0, N)   # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)   # a vector stored on the GPU filled with 2.0

y_d .+= x_d                 # add vectors together on the GPU

function add_broadcast!(y, x)
    CUDA.@sync y .+= x      # ensures the CPU waits for the GPU result to finish
    return
end
@btime add_broadcast!($y_d, $x_d)

function gpu_add!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i] # what happens if we remove @inbounds?
    end    
    return nothing
end

# fast testing of kernels
numblocks = ceil(Int, N/256)
fill!(y_d, 2)
@cuda threads=256 blocks=numblocks gpu_add!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

# launch configuration for portability
kernel = @cuda launch=false gpu_add!(y_d, x_d)
config = launch_configuration(kernel.fun)
threads = min(N, config.threads)
blocks = cld(N, threads)
fill!(y_d, 2)
kernel(y_d, x_d; threads, blocks)
@test all(Array(y_d) .== 3.0f0)

# CUDA kernels don't work well with Julia's printing. If you need 
# to print, use e.g., @cuprintln("thread $index, block $stride").