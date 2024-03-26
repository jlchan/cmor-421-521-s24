using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

x_send = nothing
if rank == 0
    x_send = collect(1 : 2 * size)
end

x_recv = MPI.bcast(x_send, comm; root = 0)

println("On rank $(rank), x_recv = $(x_recv)")

x = Vector{Int}(undef, 2 * size)
if rank == 0
    x = collect(2 * size: -1: 1)
end

MPI.Bcast!(x, comm; root = 0)
println("On rank $(rank), x = $(x)")