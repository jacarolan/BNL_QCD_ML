using LinearAlgebra
using IterativeSolvers
using BenchmarkTools

iterations = 5

n = 5000

println("The times taken to solve a random ", n , "x", n, " system.")

for i in 1:iterations
    @btime begin
        A = rand(n, n)

        A = 0.5 * (A + transpose(A)) + n * I

        b = rand(n)

        x = IterativeSolvers.cg(A, b)
    end
end