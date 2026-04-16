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
    termination_reason::Symbol
end

"""
    is_successful(result::RetroResult) -> Bool

Return `true` if the optimizer terminated because a convergence tolerance
was satisfied (`:gtol`, `:ftol`, or `:xtol`).
"""
function is_successful(result::RetroResult)
    return result.termination_reason in (:gtol, :ftol, :xtol)
end

function Base.show(io::IO, result::RetroResult{T, VT}) where {T<:Real, VT<:AbstractVector{T}}
    println(io, "RetroResult{", T, ", ", VT, "}:")
    println(io, "  Final objective value: ", result.fx)
    println(io, "  Final gradient norm:   ", norm(result.gx))
    println(io, "  Iterations:            ", result.iterations)
    println(io, "  Function evaluations:  ", result.function_evaluations)
    println(io, "  Gradient evaluations:  ", result.gradient_evaluations)
    println(io, "  Hessian evaluations:   ", result.hessian_evaluations)
    println(io, "  Termination reason:    ", result.termination_reason)
end