include("../src/Fides.jl")
using .Fides, ForwardDiff, DifferentiationInterface, Enzyme, Optimization, OptimizationOptimJL
using Profile

# Rosenbrock function with bounds
f(x) = 100*(x[2] - x[1]^2)^2 + (1 - x[1])^2

prob = FidesProblem(f, [-1.2, 1.5], AutoEnzyme(); 
                   lb=[-2.0, -2.0], ub=[2.0, 2.0])


optfunc = OptimizationFunction((x,_) -> f(x), AutoForwardDiff())
optprob = Optimization.OptimizationProblem(optfunc, [-1.2, 1.5], 
                                          lb=[-2.0, -2.0], ub=[2.0, 2.0])

# Solve with different configurations
using BenchmarkTools
@benchmark Optimization.solve($optprob, BFGS())
@benchmark Fides.solve($prob, BFGSUpdate(), TwoDimSubspace())
result1 = Fides.solve($prob, BFGSUpdate(), TwoDimSubspace())

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