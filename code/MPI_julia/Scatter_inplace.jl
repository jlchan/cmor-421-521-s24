using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

x_send = nothing
x_recv = Vector{Int}(undef, 2)
if rank == 0
    x_send = Matrix(reshape(collect(1 : 2 * size), 2, size))
    print("On rank 0, x_send = $x_send \n\n")
end

MPI.Barrier(MPI.COMM_WORLD) # just for printing

MPI.Scatter!(x_send, x_recv, comm; root = 0)

println("On rank $(rank), x_recv = $(x_recv)")
