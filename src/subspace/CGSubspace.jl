"""
    CGSubspace <: AbstractSubspace

Conjugate Gradient subspace method for large-scale problems.
Incrementally builds Krylov subspace using Hessian-vector products.

# Fields
- `max_cg_iter::Int`: Maximum CG iterations (default: min(n, 50))
- `cg_tol::Real`: CG tolerance (default: 1e-6)
"""
struct CGSubspace{T<:Real} <: AbstractSubspace
    max_cg_iter::Int
    cg_tol::T
    
    CGSubspace{T}(; max_cg_iter::Int = 50, cg_tol::T = T(1e-6)) where {T} = new{T}(max_cg_iter, cg_tol)
end

CGSubspace(; kwargs...) = CGSubspace{Float64}(; kwargs...)

mutable struct CGSubspaceState{T<:Real}
    cg_iter::Int
    residual_norm::T
    
    CGSubspaceState{T}() where {T} = new{T}(0, zero(T))
end

function init_subspace!(::CGSubspace{T}, ::RetroCache{T}) where {T}
    return CGSubspaceState{T}()
end

function build_subspace!(cg::CGSubspace{T}, state, cache::RetroCache{T}, hess_approx, hess_state, x) where {T}
    n = length(cache.g)
    
    copy!(cache.r, cache.g) 
    copy!(cache.d, cache.g) 
    fill!(cache.p, zero(T))
    
    state.residual_norm = norm(cache.r)
    
    max_iter = min(cg.max_cg_iter, n)
    
    for k in 1:max_iter
        state.cg_iter = k
        
        if state.residual_norm < cg.cg_tol
            break
        end
        
        apply_hessian!(cache.Hd, hess_approx, hess_state, cache, cache.d)
        
        dHd = dot(cache.d, cache.Hd)
        if dHd <= zero(T)
            dd = dot(cache.d, cache.d)
            τ = (-dot(cache.p, cache.d) + sqrt(dot(cache.p, cache.d)^2 + dd * (cg.cg_tol^2 - dot(cache.p, cache.p)))) / dd
            @. cache.p += τ * cache.d
            return
        end
        
        rr = dot(cache.r, cache.r)
        α = rr / dHd
        
        @. cache.p += α * cache.d
        @. cache.r += α * cache.Hd
        
        rr_new = dot(cache.r, cache.r)
        state.residual_norm = sqrt(rr_new)
        
        if k < max_iter 
            β = rr_new / rr
            @. cache.d = -cache.r + β * cache.d
        end
    end
    
    @. cache.p = -cache.p
end

function solve_subspace_tr!(solver, ::CGSubspace{T}, state, cache::RetroCache{T}, Δ::T) where {T}
    p_norm = norm(cache.p)
    
    if p_norm > Δ
        @. cache.p *= Δ / p_norm
        return Δ
    else
        return p_norm
    end
end