include("../src/Fides.jl")
using ForwardDiff, DifferentiationInterface, Optimization, OptimizationOptimJL
using .Fides: FidesProblem, BFGSUpdate, TwoDimSubspace, solve, analyze_result
using LinearAlgebra

# Rosenbrock function with bounds
f(x) = 100*(x[2] - x[1]^2)^2 + (1 - x[1])^2

prob = FidesProblem(f, [-1.2, 1.5], AutoForwardDiff(); 
                   lb=[-2.0, -2.0], ub=[2.0, 2.0])

# Test with verbose output
options = Fides.TrustRegionOptions(verbose=true, maxiter=100)
result1 = Fides.solve(prob, BFGSUpdate(), TwoDimSubspace(); options=options)

println("\n=== Final Result ===")
println("x = ", result1.x)
println("f(x) = ", result1.fx)
println("Expected x = [1.0, 1.0]")
println("Expected f(x) = 0.0")
println("Gradient norm: ", norm(result1.gx, Inf))
println("Converged: ", result1.converged)
println("Reason: ", result1.convergence_reason)

# Also try with exact Hessian to see if it's a BFGS issue
println("\n=== Testing with Exact Hessian ===")
result2 = Fides.solve(prob, Fides.ExactHessian(), TwoDimSubspace(); options=options)
println("x = ", result2.x)
println("f(x) = ", result2.fx)

# Test different subproblem solvers
println("\n=== Testing with CGSubspace ===")
result3 = Fides.solve(prob, BFGSUpdate(), Fides.CGSubspace(); options=options)
println("x = ", result3.x)
println("f(x) = ", result3.fx)

using BenchmarkTools

#@benchmark Fides.solve($prob, BFGSUpdate(), TwoDimSubspace())

# Analyze results
analyze_result(result1)
result1.x
# create a plot of the loss landscape and the found solution
using CairoMakie
xs = range(-2.0, 2.0, length=400)
ys = range(-1.0, 3.0, length=400)
Z = [f([x, y]) for x in xs, y in ys]
fig = let f = Figure(size = (300, 300))
    ax = Axis(f[1, 1], xlabel="x₁", ylabel="x₂", title="Rosenbrock Function with Bounds")
    contourf!(ax, xs, ys, -log.(Z), levels=10, colormap = :lapaz)
    scatter!(ax, [result1.x[1]], [result1.x[2]], strokewidth=2, color=:transparent, strokecolor = Makie.ColorSchemes.colorschemes[:sanzo_091][2], markersize = 12, marker=:circle, label="Found Minimum")
    #Colorbar(f[1, 2], label = "f(x)")
    f
end
save("rosenbrock_optimization.png", fig)
Makie.ColorSchemes.colorschemes[:sanzo_001]