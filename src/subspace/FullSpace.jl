"""
    FullSpace <: AbstractSubspace

Full-space trust-region method. This method constructs the full Hessian matrix and solves the trust-region subproblem in the 
original space. It is more expensive than subspace methods, but can be more accurate for small to medium-sized problems.
"""
struct FullSpace <: AbstractSubspace end

mutable struct FullSpaceState{T<:Real, M}
    H::M  
    n::Int
    
    function FullSpaceState{T}(n::Int) where {T}
        H = zeros(T, n, n)
        new{T, typeof(H)}(H, n)
    end
end

function init_subspace!(::FullSpace, cache::RetroCache{T}) where {T}
    n = length(cache.g)
    return FullSpaceState{T}(n)
end

function build_subspace!(::FullSpace, state, cache::RetroCache{T}, hess_approx, hess_state, x) where {T}
    n = state.n

    for i in 1:n
        fill!(cache.tmp, zero(T))
        cache.tmp[i] = one(T)
        
        apply_hessian!(cache.v1, hess_approx, hess_state, cache, cache.tmp)
        
        for j in 1:n
            state.H[j,i] = cache.v1[j]
        end
    end

end

function solve_subspace_tr!(::AbstractTRSolver, ::FullSpace, state, cache::RetroCache{T}, Δ::T) where {T}

    try
        copy!(cache.tmp, cache.g)
        ldiv!(factorize(state.H), cache.tmp) 
        @. cache.p = -cache.tmp 

        p_norm = norm(cache.p)
        if p_norm > Δ
            @. cache.p *= Δ / p_norm
            return Δ
        else
            return p_norm
        end
    catch
        g_norm = norm(cache.g)
        if g_norm > eps(T)
            α = min(Δ / g_norm, one(T))
            @. cache.p = -α * cache.g
            return α * g_norm
        else
            fill!(cache.p, zero(T))
            return zero(T)
        end
    end
end