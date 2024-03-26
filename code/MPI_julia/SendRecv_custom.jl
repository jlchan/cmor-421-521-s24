using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

struct Person
  age::Int64
  height::Float64
  name::String
end

p = Person(20, 1.83, "Tom")

if rank == 0
    MPI.send(p, comm; dest=1)
elseif rank > 0
    data = MPI.recv(comm; source=(rank - 1))
    MPI.send(p, comm; dest=(rank + 1) % size)
end

if rank == 0
    data = MPI.recv(comm; source=(size - 1))
end

print("My rank is $(rank)\n I received this: $(data)\n")
MPI.Barrier(comm)