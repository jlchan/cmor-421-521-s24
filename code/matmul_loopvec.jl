using BenchmarkTools 
using LoopVectorization
using LinearAlgebra

# Assumes C, A, B are matrices, not vectors
function matmul_naive!(C, A, B)
    n = size(C, 1)
    for j in 1:n
        for i in 1:n
            Cij = C[i, j]
            for k in 1:n
                Cij += A[i, k] * B[k, j]
            end
            C[i, j] = Cij
        end
    end
end

function matmul_loopvec!(C, A, B)
    n = size(C, 1)
    @turbo for i in 1:n
        for j in 1:n
            Cij = C[i, j]
            for k in 1:n
                Cij += A[i, k] * B[k, j]
            end
            C[i, j] = Cij
        end
    end
end

n = 512
A = randn(n, n)
B = randn(n, n)
C = zeros(n, n)

println("Naive matmul runtime: ")
b_naive = @btime matmul_naive!($C, $A, $B)
println("LoopVectorization matmul runtime:")
b_loopvec = @btime matmul_loopvec!($C, $A, $B)
