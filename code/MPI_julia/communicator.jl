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

x = (row_color, col_color)

row_bcast = MPI.bcast(x, row_comm; root=0)
col_bcast = MPI.bcast(x, col_comm; root=0)

# gather all the results to 
row_results = 
Gather(row_bcast, row_results, MPI.COMM_WORLD)