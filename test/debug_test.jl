include("../src/Fides.jl")
using .Fides
using LinearAlgebra
using DifferentiationInterface

# Simple quadratic: f(x) = 0.5 * x^T * x
f(x) = 0.5 * dot(x, x)

x0 = [2.0, 3.0]
prob = FidesProblem(f, x0, AutoForwardDiff())

options = Fides.TrustRegionOptions(verbose=true, maxiter=10, gtol=1e-6)
result = Fides.solve(prob, Fides.BFGSUpdate(), Fides.TwoDimSubspace(); options=options)

println("\n=== Simple Quadratic Result ===")
println("x = ", result.x)
println("f(x) = ", result.fx)
println("Expected x = [0.0, 0.0]")
println("Expected f(x) = 0.0")
println("Converged: ", result.converged)
println("Reason: ", result.convergence_reason)
