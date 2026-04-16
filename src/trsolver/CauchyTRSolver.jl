"""
    CauchyTRSolver <: AbstractTRSolver

Cauchy point trust-region solver.
Computes the steepest descent step along the negative gradient direction.
Fast but potentially less accurate than other methods.
"""
struct CauchyTRSolver <: AbstractTRSolver
end

# Solve trust-region subproblem using Cauchy point
function solve_tr!(::CauchyTRSolver, g::AbstractVector{T}, H::AbstractMatrix{T}, Delta::T, p::AbstractVector{T}) where {T}
    g_norm = norm(g)
    
    if g_norm < eps(T)
        fill!(p, zero(T))
        return zero(T)
    end
    
    gHg = dot(g, H, g)
    
    if gHg > eps(T)
        α_opt = g_norm^2 / gHg
        α = min(α_opt, Delta / g_norm)
    else
        α = Delta / g_norm
    end
    
    @. p = -α * g
    return α * g_norm
end