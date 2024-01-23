using LoopVectorization
using MuladdMacro

function matmul_flat_naive!(C, A, B, n)
    for i in 1:n
        for j in 1:n
            Cij = C[j + (i - 1) * n]
            for k in 1:n
                Aij = A[k + (i - 1) * n]
                Bij = B[j + (k - 1) * n]
                Cij += Aij * Bij
            end
            C[j + (i - 1) * n] = Cij
        end
    end
end

const BLOCK_SIZE = 16

# -O3 type optimizations:
# indexing from zero ?
# @inbounds
# @muladd
@muladd function matmul_flat_blocked!(C, A, B, n)
    for ii in 0:BLOCK_SIZE:(n - 1)
        for jj in 0:BLOCK_SIZE:(n - 1)
            for kk in 0:BLOCK_SIZE:(n - 1)
                for i in ii:(ii + BLOCK_SIZE - 1)
                    for j in jj:(jj + BLOCK_SIZE - 1)
                        @inbounds Cij = C[(j + 1) + i * n]
                        for k in kk:(kk + BLOCK_SIZE - 1)
                            @inbounds Aij = A[(k + 1) + i * n]
                            @inbounds Bij = B[(j + 1) + k * n]
                            Cij = Cij + Aij * Bij
                        end
                        @inbounds C[(j + 1) + i * n] = Cij
                    end
                end
            end
        end
    end
end

# Assumes C, A, B are matrices, not vectors
function matmul_naive!(C, A, B)
    n = size(C, 1)
    for i in 1:n
        for j in 1:n
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

using BenchmarkTools

n = 512
A = randn(n * n)
B = randn(n * n)
C = zeros(n * n)
matmul_flat_naive!(C, A, B, n)
@assert norm(reshape(C, n, n)' - reshape(A, n, n)' * reshape(B, n, n)') < n * n * eps()

C .= 0
matmul_flat_blocked!(C, A, B, n)
@assert norm(reshape(C, n, n)' - reshape(A, n, n)' * reshape(B, n, n)') < n * n * eps()

println("Naive flat matmul runtime: ")
@btime matmul_flat_naive!($A, $B, $C, $n)
println("Blocked flat matmul runtime: ")
@btime matmul_flat_blocked!($A, $B, $C, $n)

A = randn(n, n)
B = randn(n, n)
C = zeros(n, n)

println("Naive matmul runtime: ")
b_naive = @btime matmul_naive!($C, $A, $B)
# println("LoopVectorization matmul runtime:")
# b_loopvec = @btime matmul_loopvec!($C, $A, $B)
