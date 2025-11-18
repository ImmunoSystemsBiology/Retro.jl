# ============================================================================
# Abstract Types
# ============================================================================

abstract type AbstractHessianUpdate end
abstract type AbstractSubproblemSolver end

# ============================================================================
# Hessian Update Types
# ============================================================================

struct BFGSUpdate <: AbstractHessianUpdate end
struct SR1Update <: AbstractHessianUpdate end  
struct ExactHessian <: AbstractHessianUpdate end

# ============================================================================
# Subproblem Solver Types
# ============================================================================

struct TwoDimSubspace <: AbstractSubproblemSolver end
struct CGSubspace <: AbstractSubproblemSolver 
    maxiter::Int
    CGSubspace(maxiter::Int = 200) = new(maxiter)
end
struct FullSpace <: AbstractSubproblemSolver end

# ============================================================================
# Problem Definition (SciML-style)
# ============================================================================

struct FidesProblem{F, X, ADT, LB, UB}
    f::F                      # Objective function
    x0::X                     # Initial conditions
    adtype::ADT              # AD backend
    lb::LB                   # Lower bounds (nothing or vector)  
    ub::UB                   # Upper bounds (nothing or vector)
    
    function FidesProblem(f::F, x0::X, adtype::ADT; 
                         lb::LB=nothing, ub::UB=nothing) where {F, X, ADT, LB, UB}
        # Validate bounds
        if lb !== nothing && length(lb) != length(x0)
            throw(ArgumentError("Lower bounds must have same length as x0"))
        end
        if ub !== nothing && length(ub) != length(x0)
            throw(ArgumentError("Upper bounds must have same length as x0"))
        end
        if lb !== nothing && ub !== nothing
            if any(lb .>= ub)
                throw(ArgumentError("Lower bounds must be less than upper bounds"))
            end
        end
        
        new{F, X, ADT, LB, UB}(f, x0, adtype, lb, ub)
    end
end

# ============================================================================
# Algorithm Options
# ============================================================================

struct TrustRegionOptions{T<:Real}
    # Convergence tolerances
    gtol::T                    # Gradient tolerance
    xtol::T                    # Step tolerance  
    ftol::T                    # Function tolerance
    
    # Trust region parameters
    initial_tr_radius::T       # Initial trust region radius
    max_tr_radius::T          # Maximum trust region radius
    eta1::T                   # Shrink threshold
    eta2::T                   # Expand threshold
    gamma1::T                 # Shrink factor
    gamma2::T                 # Expand factor
    
    # Reflective bounds parameters
    theta1::T                 # Reflection threshold 1
    theta2::T                 # Reflection threshold 2
    
    # Algorithm parameters
    maxiter::Int              # Maximum iterations
    
    # Miscellaneous
    verbose::Bool
    
    function TrustRegionOptions{T}(;
        gtol::T = T(1e-6),
        xtol::T = T(0.0), 
        ftol::T = T(1e-9),
        initial_tr_radius::T = T(1.0),
        max_tr_radius::T = T(1000.0),
        eta1::T = T(0.25),
        eta2::T = T(0.75),
        gamma1::T = T(0.25),
        gamma2::T = T(2.0),
        theta1::T = T(0.1),
        theta2::T = T(0.2), 
        maxiter::Int = 1000,
        verbose::Bool = false
    ) where {T<:Real}
        new{T}(gtol, xtol, ftol, initial_tr_radius, max_tr_radius,
               eta1, eta2, gamma1, gamma2, theta1, theta2, maxiter, verbose)
    end
end

TrustRegionOptions(; kwargs...) = TrustRegionOptions{Float64}(; kwargs...)

# ============================================================================
# Result Structure
# ============================================================================

struct TrustRegionResult{T<:Real, VT<:AbstractVector{T}}
    x::VT                     # Final solution
    fx::T                     # Final function value
    gx::VT                    # Final gradient
    iterations::Int           # Number of iterations
    function_evaluations::Int # Function evaluation count
    gradient_evaluations::Int # Gradient evaluation count
    hessian_evaluations::Int  # Hessian evaluation count
    converged::Bool          # Convergence flag
    convergence_reason::Symbol # Reason for termination
end

# ============================================================================
# Internal State
# ============================================================================

mutable struct TrustRegionState{T<:Real, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}}
    x::VT                     # Current iterate
    fx::T                     # Function value at x
    gx::VT                    # Gradient at x
    gx_free::VT              # Gradient projected onto free variables
    Hx::MT                    # Hessian/approximation at x
    tr_radius::T             # Trust region radius
    
    # Bound constraints
    lb::Union{VT, Nothing}   # Lower bounds
    ub::Union{VT, Nothing}   # Upper bounds
    active_set::BitVector    # Active constraints
    
    # Iteration counters
    iter::Int
    f_evals::Int
    g_evals::Int
    h_evals::Int
    
    # Workspace vectors
    step::VT
    step_reflected::VT
    x_trial::VT
    g_trial::VT
    Hg::VT
    
    function TrustRegionState(x0::VT, fx0::T, gx0::VT, Hx0::MT, tr_radius::T,
                             lb::Union{VT, Nothing}, ub::Union{VT, Nothing}) where {T<:Real, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}}
        n = length(x0)
        step = similar(x0)
        step_reflected = similar(x0)
        x_trial = similar(x0)
        g_trial = similar(x0)
        gx_free = similar(x0)
        Hg = similar(x0)
        active_set = BitVector(undef, n)
        
        new{T, VT, MT}(copy(x0), fx0, copy(gx0), gx_free, copy(Hx0), tr_radius,
                       lb === nothing ? nothing : copy(lb),
                       ub === nothing ? nothing : copy(ub),
                       active_set, 0, 1, 1, 0, step, step_reflected, x_trial, g_trial, Hg)
    end
end