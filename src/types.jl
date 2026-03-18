using DifferentiationInterface
using ADTypes: AbstractADType
using LinearAlgebra
using StaticArrays

# Core abstract types

"""
    AbstractObjectiveFunction

Supertype for all Retro objective functions.

Concrete subtypes: [`ADObjectiveFunction`](@ref), [`GradientObjectiveFunction`](@ref),
[`AnalyticObjectiveFunction`](@ref).
"""
abstract type AbstractObjectiveFunction end

"""
    AbstractSubspace

Supertype for subspace strategies used to reduce the trust-region subproblem.

Concrete subtypes: [`TwoDimSubspace`](@ref), [`CGSubspace`](@ref),
[`FullSpace`](@ref).
"""
abstract type AbstractSubspace end

"""
    AbstractTRSolver

Supertype for trust-region subproblem solvers.

Concrete subtypes: [`EigenTRSolver`](@ref), [`CauchyTRSolver`](@ref).
"""
abstract type AbstractTRSolver end

"""
    AbstractHessianApproximation

Supertype for Hessian approximation strategies.

Every subtype must implement [`init_hessian!`](@ref), [`update_hessian!`](@ref),
and [`apply_hessian!`](@ref).

Optionally implement [`reset_hessian!`](@ref) to support resetting the
approximation when the optimizer stagnates.

Concrete subtypes: [`BFGS`](@ref), [`SR1`](@ref), [`ExactHessian`](@ref).
"""
abstract type AbstractHessianApproximation end

"""
    reset_hessian!(approx, state, cache)

Reset the Hessian approximation.  Default is a no-op;
subtypes can override for smarter recovery on stagnation.
"""
reset_hessian!(::AbstractHessianApproximation, state, cache) = nothing

"""
    AbstractDisplayMode

Supertype for output verbosity levels.

Concrete subtypes: [`Silent`](@ref), [`Iteration`](@ref), [`Final`](@ref),
[`Verbose`](@ref).
"""
abstract type AbstractDisplayMode end

# Input types
const InputVector{T} = Union{AbstractVector{T}, StaticVector{N,T}} where {N,T}