function matmul_flat_naive!(C, A, B, n)
    for i in 0:n-1
        for j in 0:n-1
            Cij = C[j + 1 + i * n]
            for k in 0:n-1
                Aij = A[k + 1 + i * n]
                Bij = B[j + 1 + k * n]
                Cij += Aij * Bij
            end
            C[j + 1 + i * n] = Cij
        end
    end
end

# Similar to #define BLOCK_SIZE 16
const BLOCK_SIZE = 16

using MuladdMacro

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

using LinearAlgebra

n = 512
A = randn(n * n)
B = randn(n * n)
C = zeros(n * n)
@time matmul_flat_naive!(C, A, B, n)
@assert norm(reshape(C, n, n)' - reshape(A, n, n)' * reshape(B, n, n)') < n * n * eps()

C .= 0
@time matmul_flat_blocked!(C, A, B, n)
@assert norm(reshape(C, n, n)' - reshape(A, n, n)' * reshape(B, n, n)') < n * n * eps()

using BenchmarkTools

println("Naive flat matmul runtime: ")
@btime matmul_flat_naive!($A, $B, $C, $n)
println("Blocked flat matmul runtime: ")
@btime matmul_flat_blocked!($A, $B, $C, $n)

