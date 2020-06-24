using LinearAlgebra
using IterativeSolvers
using Plots

n = 5000

A = rand(n, n)

A = 0.5 * (A + transpose(A)) + n * I

b = rand(n)

x, ch = IterativeSolvers.cg(A, b; log=true)


resNorms = ch[:resnorm]

plot(1:ch.iters, resNorms, fmt=:png)
plot!(title = "Residual vs. iteration", xlabel = "Iteration", ylabel = "Residual Error")

# or using magic:
plot!(yaxis = ("Residual Error", :log10))
png(string("Error", n, "x", n, ".png"))