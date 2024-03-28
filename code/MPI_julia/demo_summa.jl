using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

p = Int(sqrt(size))
@assert p * p == size

row_color = rank ÷ p
col_color = rank % p
row_comm = MPI.Comm_split(MPI.COMM_WORLD, row_color, rank)
col_comm = MPI.Comm_split(MPI.COMM_WORLD, col_color, rank)

x = rank #(row_color, col_color)

original_ranks = MPI.Gather(rank, MPI.COMM_WORLD; root=0)
if rank==0
    println("\n Ranks")
    show(stdout, "text/plain", Matrix(reshape(original_ranks, p, p))) 
end

for k = 0:p-1 # remember MPI is 0-indexed
    row_bcast = MPI.bcast(x, row_comm; root=k)
    col_bcast = MPI.bcast(x, col_comm; root=k)
    
    # gather all the results to 
    row_results = MPI.Gather(row_bcast, MPI.COMM_WORLD; root=0)
    col_results = MPI.Gather(col_bcast, MPI.COMM_WORLD; root=0)
    if rank==0        
        println("\nOn iteration $k, broadcasting across rows:")
        show(stdout, "text/plain", Matrix(reshape(row_results, p, p)))        

        println("\nOn iteration $k, broadcasting across cols")
        show(stdout, "text/plain", Matrix(reshape(col_results, p, p)))
        println("\n")
    end
end

