using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

# p = Int(sqrt(size))
# @assert p * p == size

# row_color = rank ÷ p
# col_color = rank % p

# original_ranks = MPI.Gather(rank, MPI.COMM_WORLD; root=0)
# if rank==0
#     println("\n Ranks")
#     show(stdout, "text/plain", Matrix(reshape(original_ranks, p, p))) 
# end

send = [rank]
recv = [rank]
if rank==0
    MPI.Sendrecv!(send, recv, MPI.COMM_WORLD; source=size-1, dest=1)
else
    MPI.Sendrecv!(send, recv, MPI.COMM_WORLD; source=rank-1, dest=(rank + 1) % size)
end

# for k = 0:p-1 # remember MPI is 0-indexed
#     col_bcast = MPI.bcast(x, col_comm; root=k)
    
#     # gather all the results to 
#     row_results = MPI.Gather(row_bcast, MPI.COMM_WORLD; root=0)
#     col_results = MPI.Gather(col_bcast, MPI.COMM_WORLD; root=0)
#     if rank==0        
#         println("\nOn iteration $k, broadcasting across rows:")
#         show(stdout, "text/plain", Matrix(reshape(row_results, p, p)))        

#         println("\nOn iteration $k, broadcasting across cols")
#         show(stdout, "text/plain", Matrix(reshape(col_results, p, p)))
#         println("\n")
#     end
# end

