module Fides

using LinearAlgebra
using DifferentiationInterface
using ForwardDiff  # For fallback AD backend

include("types.jl")
include("bounds.jl")
include("subproblem.jl")
include("hessian.jl")
include("solve.jl")

export FidesProblem, solve, TrustRegionOptions
export BFGSUpdate, SR1Update, ExactHessian
export TwoDimSubspace, CGSubspace, FullSpace
export analyze_result

# ============================================================================
# Extended Result Analysis
# ============================================================================

"""
    analyze_result(result::TrustRegionResult)

Provide detailed analysis of optimization result.
"""
function analyze_result(result::TrustRegionResult)
    println("=== Fides Optimization Result ===")
    println("Converged: $(result.converged)")
    println("Convergence reason: $(result.convergence_reason)")
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