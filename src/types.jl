"""
    Abstract Types

Base types for the Retro optimization framework.
"""
abstract type AbstractHessianUpdate end
abstract type ExactHessianUpdate <: AbstractHessianUpdate end
abstract type ApproximatingHessianUpdate <: AbstractHessianUpdate end

"""
    NullParameters

Singleton type indicating that the objective function does not use parameters.
When `NullParameters()` is passed to `solve`, the objective function `f(x, p)`
is called as `f(x, NullParameters())`, allowing the function to ignore the
parameter argument.
"""
struct NullParameters end

abstract type AbstractSubspace end
abstract type AbstractSubproblemSolver end

"""
    BFGSUpdate <: ApproximatingHessianUpdate

Broyden-Fletcher-Goldfarb-Shanno (BFGS) Hessian approximation.

BFGS is a rank-2 quasi-Newton update that maintains a positive-definite Hessian
approximation. It's a generally robust choice for unconstrained and
bound-constrained optimization.
"""
struct BFGSUpdate <: ApproximatingHessianUpdate end

"""
    SR1Update <: ApproximatingHessianUpdate

Symmetric Rank-1 (SR1) Hessian approximation.

SR1 is a rank-1 quasi-Newton update that can better approximate indefinite Hessians
but may lose positive-definiteness. Good for problems where the true Hessian is
indefinite or ill-conditioned.
"""
struct SR1Update <: ApproximatingHessianUpdate end

"""
    ExactHessian <: ExactHessianUpdate

Use exact Hessian computed via automatic differentiation.

Computes the full Hessian matrix at each iteration using second-order AD.
Most accurate but computationally expensive. Best for small-dimensional problems
where accuracy is critical.
"""
struct ExactHessian <: ExactHessianUpdate end

"""
    GaussNewtonUpdate <: ExactHessianUpdate

Gauss-Newton Hessian approximation for least-squares problems.

For objectives of the form `f(x) = 0.5 * ||r(x)||²`, approximates the Hessian as
`H ≈ J'*J` where J is the Jacobian of the residual vector r(x).

This is particularly effective when residuals are small near the solution.
Requires the user to provide a function that returns residual vector.

# Fields
- `resfun`: Function that computes residuals: `r = resfun(x, p)`

# Example
```julia
# Residual function
residuals(x, p) = x .- p.target
# Objective function (for other methods)
objective(x, p) = 0.5 * sum(abs2, residuals(x, p))
# Create problem with objective
prob = RetroProblem(objective, x0, AutoForwardDiff())
# Use Gauss-Newton with residual function
result = solve(prob, GaussNewtonUpdate(residuals), TwoDimSubspace(), p)
```
"""
struct GaussNewtonUpdate{F} <: ExactHessianUpdate
    resfun::F
end

"""
    EigenvalueSolver <: AbstractSubproblemSolver

Solve trust-region subproblem using eigenvalue decomposition (exact solution).

Solves the problem:
```math
\\min_s \\{s^T B s + s^T g : ||s|| \\leq \\Delta\\}
```

The solution is characterized by `-(B + λI)s = g`. If B is positive definite,
the solution can be obtained by `λ = 0` if `Bs = -g` satisfies `||s|| ≤ Δ`.
Otherwise, an appropriate `λ` is identified via 1D rootfinding of the secular equation:

```math
\\phi(\\lambda) = \\frac{1}{||s(\\lambda)||} - \\frac{1}{\\Delta} = 0
```

The eigenvalue decomposition has the advantage that eigenvectors are invariant
to changes in `λ` and eigenvalues are linear in `λ`, so factorization only needs
to be performed once. We use Newton's method with ITP bracketing solver as fallback.
The hard case is treated separately.

# Fields
- `max_newton_iterations::Int`: Maximum Newton iterations for secular equation (default: 50)
- `newton_tolerance::Float64`: Tolerance for Newton solver (default: 1e-12)
"""
struct EigenvalueSolver <: AbstractSubproblemSolver 
    max_newton_iterations::Int
    newton_tolerance::Float64
    function EigenvalueSolver(; max_newton_iterations::Int=50, newton_tolerance::Float64=1e-12)
        new(max_newton_iterations, newton_tolerance)
    end
end

"""
    CauchyPointSolver <: AbstractSubproblemSolver

Solve trust-region subproblem using Cauchy point (fast approximate solution).

Computes the Cauchy point, which is the minimizer along the steepest descent
direction within the trust region. This is much faster than the exact eigenvalue
solution but may be less accurate.

The Cauchy point is defined as:
```math
s_c = -t \\cdot g, \\quad t = \\min\\left(\\frac{||g||^2}{g^T B g}, \\frac{\\Delta}{||g||}\\right)
```

This solver is recommended for large-scale problems where speed is more important
than finding the exact subproblem solution.
"""
struct CauchyPointSolver <: AbstractSubproblemSolver end


"""
    TwoDimSubspace <: AbstractSubspace

Solve trust-region subproblem in a 2D subspace.

Projects the problem onto a 2-dimensional subspace spanned by the gradient and
Newton directions, then solves exactly in that subspace. Good balance between
speed and quality for most problems.

# Fields
- `solver::AbstractSubproblemSolver`: Solver to use for the 2D subproblem
  - `EigenvalueSolver()`: Exact solution (default, most accurate)
  - `CauchyPointSolver()`: Fast approximate solution (faster but less accurate)
"""
struct TwoDimSubspace <: AbstractSubspace 
    solver::AbstractSubproblemSolver
    function TwoDimSubspace(; solver::AbstractSubproblemSolver=EigenvalueSolver())
        new(solver)
    end
end

"""
    FullSpace <: AbstractSubspace

Solve full-dimensional trust-region subproblem.

Solves the subproblem in the full space using eigenvalue decomposition.
Most accurate but expensive for large problems. Always uses EigenvalueSolver.

# Fields
- `solver::AbstractSubproblemSolver`: Always EigenvalueSolver for full space problems
"""
struct FullSpace <: AbstractSubspace 
    solver::AbstractSubproblemSolver
    function FullSpace(; solver::AbstractSubproblemSolver=EigenvalueSolver())
        if !(solver isa EigenvalueSolver)
            @warn "FullSpace only supports EigenvalueSolver. Using EigenvalueSolver()."
            return new(EigenvalueSolver())
        end
        new(solver)
    end
end

"""
    CGSubspace([maxiter]) <: AbstractSubspace

Solve trust-region subproblem using conjugate gradient (Steihaug-Toint).

Uses truncated conjugate gradient to solve the subproblem. Efficient for large
problems where forming the full Hessian or 2D subspace is expensive.

# Arguments
- `maxiter::Int`: Maximum CG iterations (default: 200)
"""
struct CGSubspace <: AbstractSubspace
    maxiter::Int
    CGSubspace(maxiter::Int=200) = new(maxiter)
end




"""
    RetroProblem{F, X, ADT, LB, UB}

Define an optimization problem for Retro.

Encapsulates an objective function, initial point, automatic differentiation backend,
and optional bound constraints.

# Fields
- `f::F`: Objective function to minimize, in the form `f(x, p)` where `x` is the variable to be optimized and `p` are any fixed parameters.
- `x0::X`: Initial guess for optimization variables
- `adtype::ADT`: Automatic differentiation backend (e.g., `AutoForwardDiff()`)
- `lb::X`: Lower bounds (-Inf or vector matching `x0`)
- `ub::X`: Upper bounds (Inf or vector matching `x0`)

# Example
    ```julia
    f(x) = sum(abs2, x - [1.0, 2.0])
    prob = RetroProblem(f, [0.0, 0.0], AutoForwardDiff(); lb=[-10.0, -10.0], ub=[10.0, 10.0])
    ```
"""
struct RetroProblem{F, X, ADT}
    f::F
    x0::X
    adtype::ADT
    lb::X
    ub::X
    
    function RetroProblem(f::F, x0::X, adtype::ADT; 
                         lb::X=fill(eltype(x0)(-Inf), length(x0)), 
                         ub::X=fill(eltype(x0)(Inf), length(x0))) where {F, X, ADT}

        # validate bounds
        if length(lb) != length(x0)
            throw(ArgumentError("Length of lower bounds must match length of x0"))
        end

        if length(ub) != length(x0)
            throw(ArgumentError("Length of upper bounds must match length of x0"))
        end

        if any(lb .>= ub)
            throw(ArgumentError("Each lower bound must be less than the corresponding upper bound"))
        end
        
        new{F, X, ADT}(f, x0, adtype, lb, ub)
    end
end

"""
    RetroOptions{T<:Real}

Algorithm parameters for trust-region optimization.

# Convergence Criteria
- `xtol::T`: Step tolerance (default: 0.0, disabled)
- `ftol_a::T`: Absolute function tolerance (default: 1e-8)
- `ftol_r::T`: Relative function tolerance (default: 1e-8
- `gtol_a::T`: Absolute gradient tolerance (default: 1e-6)
- `gtol_r::T`: Relative gradient tolerance (default: 0.0, disabled)

# Trust Region Parameters  
- `initial_tr_radius::T`: Initial trust region radius (default: 1.0)
- `max_tr_radius::T`: Maximum allowed radius (default: 1000.0)
- `mu::T`: Shrink threshold - shrink if ρ < mu (default: 0.25)
- `eta::T`: Expand threshold - expand if ρ > eta (default: 0.75)
- `gamma1::T`: Shrink factor (default: 0.25)
- `gamma2::T`: Expand factor (default: 2.0)

# Bound Constraint Parameters
- `theta1::T`: Reflection threshold for bounds (default: 0.1)
- `theta2::T`: Secondary reflection threshold (default: 0.2)

# Example
```julia
opts = RetroOptions(gtol_a=1e-6, maxiter=100, verbose=true)
```
"""
struct RetroOptions{T<:Real}
    # Convergence tolerances
    xtol::T
    ftol_a::T
    ftol_r::T
    gtol_a::T
    gtol_r::T
    
    # Trust region parameters
    initial_tr_radius::T
    max_tr_radius::T
    mu::T
    eta::T
    gamma1::T
    gamma2::T
    
    # Reflective bounds parameters
    theta1::T
    theta2::T
    
    function RetroOptions{T}(;
        xtol::T = T(0.0), 
        gtol_a::T = T(1e-6),
        gtol_r::T = T(0.0),
        ftol_a::T = T(1e-8),
        ftol_r::T = T(1e-8),
        initial_tr_radius::T = T(1.0),
        max_tr_radius::T = T(1000.0),
        mu::T = T(0.25),
        eta::T = T(0.75),
        gamma1::T = T(0.25),
        gamma2::T = T(2.0),
        theta1::T = T(0.1),
        theta2::T = T(0.2)
    ) where {T<:Real}
        new{T}(xtol, ftol_a, ftol_r, gtol_a, gtol_r, initial_tr_radius, max_tr_radius,
               mu, eta, gamma1, gamma2, theta1, theta2)
    end
end

RetroOptions(; kwargs...) = RetroOptions{Float64}(; kwargs...)

"""
    RetroResult{T<:Real, VT<:AbstractVector{T}}

Results from trust-region optimization.

# Fields
- `x::VT`: Final solution
- `fx::T`: Final objective function value
- `gx::VT`: Final gradient
- `iterations::Int`: Number of iterations performed
- `function_evaluations::Int`: Total function evaluations
- `gradient_evaluations::Int`: Total gradient evaluations
- `hessian_evaluations::Int`: Total Hessian evaluations
- `converged::Bool`: Whether optimization converged
- `termination_reason::Symbol`: Reason for termination
  - `:gtol`: Gradient tolerance satisfied
  - `:ftol`: Function tolerance satisfied
  - `:xtol`: Step tolerance satisfied
  - `:maxiter`: Maximum iterations reached
  - `:stagnation`: Too many rejected steps
  - `:tr_radius_too_small`: Trust region became too small
"""
struct RetroResult{T<:Real, VT<:AbstractVector{T}}
    x::VT
    fx::T
    gx::VT
    iterations::Int
    function_evaluations::Int
    gradient_evaluations::Int
    hessian_evaluations::Int
    converged::Bool
    termination_reason::Symbol
end

"""
    TrustRegionState{T, VT, MT}

Internal state for trust-region optimization (not exported).

Maintains all the working variables, buffers, and counters needed during optimization.
Uses DiffResults for efficient computation and storage of function values, gradients,
and Hessians.
"""
mutable struct TrustRegionState{T<:Real, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}}
    # Current iterate
    x::VT
    
    # Primal, gradient, and hessian buffers
    prep # prep object for AD
    value::T
    grad::VT
    hessian::MT
    
    tr_radius::T
    
    # Bound constraints
    lb::VT
    ub::VT
    active_set::BitVector
    gx_free::VT
    
    # Affine scaling for bounds (Coleman-Li)
    v::VT
    dv::VT
    
    # Counters
    iter::Int
    f_evals::Int
    g_evals::Int
    h_evals::Int
    
    # Workspace arrays
    step::VT
    step_reflected::VT
    Hg::VT
    Hs::VT
    Δg::VT
    last_step_norm::T
    predicted_reduction::T  # Model reduction from subproblem
    params::Any  # Parameters for objective function
    
    # Trial point buffers (to reduce allocations)
    x_trial::VT
    fx_trial::Base.RefValue{T}
    grad_trial::VT
end

function TrustRegionState(
    x0::VT,
    prep,
    value::T,
    grad::VT,
    hessian::MT,
    tr_radius::T,
    lb::VT,
    ub::VT,
    params = NullParameters()
) where {T<:Real, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}}
    n = length(x0)
    
    # Workspace
    step = zeros(T, n)
    step_reflected = zeros(T, n)
    gx_free = similar(x0)
    Hs = similar(x0)
    Hg = similar(x0)
    Δg = similar(x0)
    active_set = falses(n)
    
    # Affine scaling vectors
    v = ones(T, n)
    dv = zeros(T, n)
    
    # Trial point buffers
    x_trial = similar(x0)
    fx_trial = Ref(zero(T))
    grad_trial = similar(x0)
    
    TrustRegionState{T, VT, MT}(
        copy(x0),
        prep,
        value,
        copy(grad),
        copy(hessian),
        tr_radius,
        lb, ub, active_set, gx_free,
        v, dv,  # Affine scaling
        0, 1, 1, 0,  # Counters
        step, step_reflected, Hg,
        Hs, Δg, zero(T), zero(T),  # last_step_norm, predicted_reduction
        params,  # Parameters
        x_trial, fx_trial, grad_trial  # Trial point buffers
    )
end