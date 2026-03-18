module Retro

using Reexport: @reexport
using ADTypes: ADTypes, AbstractADType, AutoForwardDiff
using DifferentiationInterface: DifferentiationInterface, prepare_gradient, prepare_hessian
using LinearAlgebra: LinearAlgebra, I, Symmetric, cholesky, cond, dot, eigen, factorize, issuccess, ldiv!, mul!, norm
using Printf: Printf, @printf, @sprintf
using ProgressMeter: ProgressMeter, Progress
using StaticArrays: StaticArrays, @SMatrix, @SVector, SMatrix, SVector, StaticVector

@reexport using ADTypes
@reexport using ForwardDiff

# Core types and cache must come first
include("types.jl")
export AbstractObjectiveFunction, AbstractSubspace, AbstractTRSolver, 
       AbstractHessianApproximation, AbstractDisplayMode

include("problem/RetroCache.jl")
export RetroCache

# Then objective functions (which depend on cache)
include("objective.jl")
export ADObjectiveFunction, GradientObjectiveFunction, AnalyticObjectiveFunction,
       objfunc!, gradient!, hessian!, value_and_gradient!, value_gradient_and_hessian!

# Problem definition
include("problem/RetroProblem.jl")
export RetroProblem

include("problem/RetroResult.jl")
export RetroResult, is_successful

# Hessian approximations
include("hessian/BFGS.jl")
include("hessian/SR1.jl")
include("hessian/ExactHessian.jl")
export BFGS, SR1, ExactHessian, init_hessian!, update_hessian!, apply_hessian!

# Trust-region solvers (must come before subspace methods that dispatch on them)
include("trsolver/EigenTRSolver.jl")
include("trsolver/CauchyTRSolver.jl")
export EigenTRSolver, CauchyTRSolver, solve_tr!

# Subspace methods
include("subspace/TwoDimSubspace.jl")
include("subspace/CGSubspace.jl")
include("subspace/FullSpace.jl")
export TwoDimSubspace, CGSubspace, FullSpace, init_subspace!, build_subspace!, solve_subspace_tr!

# Step computation and acceptance
include("steps/Reflection.jl")
export compute_scaling!, scale_gradient!, compute_affine_scaling!,
       apply_reflective_bounds!, project_bounds!,
       initialize_away_from_bounds!, find_step_to_bound, compute_cauchy_boundary_point!

include("steps/StepAcceptance.jl")
export predicted_reduction, actual_reduction, accept_step, update_trust_region_radius,
       check_convergence, compute_cauchy_step!

include("steps/TrustRegionStep.jl")
export compute_trust_region_step!, compute_hv_product!, check_negative_curvature,
       assess_model_quality

# Utilities
include("utils/LinearAlgebraHelpers.jl") 
include("utils/Norms.jl")
include("utils/Displays.jl")
export safe_norm, safe_dot, safe_cond, is_positive_definite, smallest_eigenvalue,
       norm_inf, norm_weighted, norm_relative, norm_scaled, trust_region_norm, gradient_norm,
       Silent, Iteration, Final, Verbose, Debug, display_header, display_iteration, display_final

# Main optimization routine
include("optimize.jl")
export RetroOptions, optimize

end # module Retro