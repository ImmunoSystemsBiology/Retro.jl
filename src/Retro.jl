"""
    Retro

A high-performance trust-region optimization package for Julia.

Retro implements interior-point trust-region reflective algorithms for bound-constrained
optimization, with a focus on speed and efficiency for ODE parameter estimation problems.

Key features:
- Multiple Hessian approximation strategies (BFGS, SR1, Exact)
- Flexible subproblem solvers (2D subspace, CG, Full space)
- Automatic differentiation support via DifferentiationInterface
- Bound constraint handling with reflective boundaries
- Specialized support for ODE parameter estimation problems

Example:
```julia
using Retro

f(x) = sum(abs2, x)
prob = RetroProblem(f, [1.0, 2.0], AutoForwardDiff())
result = solve(prob, BFGSUpdate(), TwoDimSubspace())
```

For ODE parameter estimation:
```julia
prob, hess, subprob, opts = setup_ode_optimization(nll, x0, AutoForwardDiff())
result = solve(prob, hess, subprob; options=opts)
```
"""
module Retro

using LinearAlgebra
using DifferentiationInterface
using DifferentiationInterface: Constant
using Printf
import ForwardDiff
import BracketingNonlinearSolve as NLS


include("types.jl")
include("bounds.jl")
include("subproblem.jl")
include("hessian.jl")
include("gauss_newton.jl")
include("solve.jl")

export RetroProblem, solve, RetroOptions, NullParameters
export BFGSUpdate, SR1Update, ExactHessian, GaussNewtonUpdate, HybridUpdate
export TwoDimSubspace, CGSubspace, FullSpace
export EigenvalueSolver, CauchyPointSolver
export analyze_result

using Reexport
@reexport using DifferentiationInterface: AutoForwardDiff, AutoReverseDiff, AutoZygote

"""
    analyze_result(result::RetroResult)

Provide detailed analysis of an optimization result.

# Arguments
- `result::RetroResult`: The optimization result to analyze

# Prints
- Convergence status and reason
- Final objective value and gradient norm
- Iteration and evaluation counts
- Efficiency metrics
"""
function analyze_result(result::RetroResult)
    println("=== Retro Optimization Result ===")
    println("Converged: $(result.converged)")
    println("Convergence reason: $(result.termination_reason)")
    println("Final objective value: $(result.fx)")
    println("Final gradient norm (∞): $(norm(result.gx, Inf))")
    println("Iterations: $(result.iterations)")
    println("Function evaluations: $(result.function_evaluations)")
    println("Gradient evaluations: $(result.gradient_evaluations)")  
    println("Hessian evaluations: $(result.hessian_evaluations)")
    
    efficiency = result.iterations > 0 ? result.function_evaluations / result.iterations : 0.0
    println("Average f-evals per iteration: $(round(efficiency, digits=2))")
end

end # module