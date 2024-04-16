using KernelAbstractions
using CUDA

@kernel function mul_by_2_kernel!(A)
    I = @index(Global)
    A[I] = 2 * A[I]
    # no return statements allowed in KA.jl kernels
end

# run on a CPU
dev = CPU()
A = ones(1024, 1024)
mul_by_2_kernel!(dev, 64)(A, ndrange=size(A)) # ndrange = CUDA grid 
KernelAbstractions.synchronize(dev)
all(A .== 2.0)

# switch to CUDA backend
A = CuArray(ones(1024, 1024))
backend = get_backend(A)
mul_by_2_kernel!(backend, 64)(A, ndrange=size(A))
KernelAbstractions.synchronize(backend)
all(A .== 2.0)

@kernel function foo!(b, x)
    blockid = @index(Group, Linear)
    tid = @index(Local, Linear)
    i = @index(Global)
    num_threads = @groupsize()
    
    l_x = @localmem eltype(x) (num_threads) # shared memory
    p_b = @private eltype(x) (1)            # register memory

    @inbounds begin
        if i <= length(x)
            l_x[tid] = x[i]
        end
        p_b = b[i]

        @synchronize

        p_b += 2 * l_x[tid] + 1

        b[i] = p_b
    end
end

b = CuArray(zeros(64))
x = CuArray(ones(64))

num_threads = 32
n = length(x)
num_blocks = ceil(Int, n / num_threads)
backend = get_backend(x)
foo!(backend, num_threads)(b, x, ndrange=length(x))
KernelAbstractions.synchronize(backend)


