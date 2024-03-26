using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

send_buf = Array{Float64}(undef, 10)
recv_buf = Array{Float64}(undef, 10)

fill!(send_buf, Float64(rank))

if rank == 0
    MPI.Send(send_buf, comm; dest=1)
elseif rank > 0
    MPI.Recv!(recv_buf, comm; source=(rank - 1))
    MPI.Send(send_buf, comm; dest=(rank + 1) % size)
end

if rank == 0
    MPI.Recv!(recv_buf, comm; source=(size - 1))
end

print("Rank $(rank) received $(recv_buf)\n")
MPI.Barrier(comm)