using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

struct Point{T}
    x::T
    y::T
end

import Base: + 
function +(a::Point{T}, b::Point{T}) where {T}
    return Point{T}(a.x + b.x, a.y + b.y)
end
x = Point(1, 2)

# x = collect((1:size) .+ rank)
x_recv = MPI.Reduce(x, +, comm; root = 0)

if rank==0
    println("On rank $(rank), x = $(x_recv)")
end