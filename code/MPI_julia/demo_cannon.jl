using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

p = Int(sqrt(size))
@assert p * p == size

row_index = rank % p
col_index = rank ÷ p

original_ranks = MPI.Gather(rank, MPI.COMM_WORLD; root=0)
if rank==0
    println("Ranks")
    show(stdout, "text/plain", Matrix(reshape(original_ranks, p, p))) 
    println("\n")
end
MPI.Barrier(MPI.COMM_WORLD)

rank_left  = row_index + (col_index + 2 * p - 1) * p % size
rank_right = row_index + (col_index + 1) * p % size
rank_up   = (row_index + 2 * p - 1) % p + (col_index * p)
rank_down = (row_index + 1) % p         + (col_index * p)

send = [rank]
recv = [rank]
MPI.Sendrecv!(send, recv, MPI.COMM_WORLD; source=rank_right, dest=rank_left)

left_shifted_ranks = MPI.Gather(recv[1], MPI.COMM_WORLD; root=0)
if rank==0
    println("Left shifted ranks")
    show(stdout, "text/plain", Matrix(reshape(left_shifted_ranks, p, p))) 
    println("\n")
end

send = [rank]
recv = [rank]
MPI.Sendrecv!(send, recv, MPI.COMM_WORLD; source=rank_down, dest=rank_up)

up_shifted_ranks = MPI.Gather(recv[1], MPI.COMM_WORLD; root=0)
if rank==0
    println("Up shifted ranks")
    show(stdout, "text/plain", Matrix(reshape(up_shifted_ranks, p, p))) 
    println("\n")
end
