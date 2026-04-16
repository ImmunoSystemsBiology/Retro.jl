"""
    TwoDimSubspace <: AbstractSubspace

Two-dimensional subspace spanned by gradient direction and Newton/curvature direction.
Uses StaticArrays for zero-allocation computations in the 2D subspace.

# Fields
- `normalize::Bool`: Whether to normalize basis vectors (default: true)
"""
struct TwoDimSubspace <: AbstractSubspace
    normalize::Bool
    
    TwoDimSubspace(; normalize::Bool = true) = new(normalize)
end

mutable struct TwoDimSubspaceState{T<:Real}

    g2d::SVector{2,T} 
    H2d::SMatrix{2,2,T,4} 
    p2d::SVector{2,T}  

    v1_norm::T
    v2_norm::T
    dimension::Int 
    
    function TwoDimSubspaceState{T}() where {T}
        new{T}(
            @SVector(zeros(T, 2)),
            @SMatrix(zeros(T, 2, 2)),
            @SVector(zeros(T, 2)),
            zero(T), zero(T), 0
        )
    end
end

"""
    init_subspace!(subspace, cache) -> state

Initialise the subspace method, returning an opaque `state` object.
"""
function init_subspace!(::TwoDimSubspace, ::RetroCache{T}) where {T}
    return TwoDimSubspaceState{T}()
end

"""
    build_subspace!(subspace, state, cache, hess_approx, hess_state, x)

Build the subspace basis vectors and the reduced gradient / Hessian for
the current iterate `x`.
"""
function build_subspace!(subspace::TwoDimSubspace, state, cache::RetroCache{T}, hess_approx, hess_state, x) where {T}
    n = length(cache.g)
    
    copy!(cache.v1, cache.g)
    state.v1_norm = norm(cache.v1)
    
    if state.v1_norm < eps(T)
        state.dimension = 0
        return
    end
    
    if subspace.normalize
        @. cache.v1 /= state.v1_norm
    end

    solve_newton_direction!(cache.v2, hess_approx, hess_state, cache, cache.g)
    state.v2_norm = norm(cache.v2)
    
    v1_dot_v2 = dot(cache.v1, cache.v2)
    @. cache.v2 -= v1_dot_v2 * cache.v1
    
    v2_norm_ortho = norm(cache.v2)
    
    if v2_norm_ortho < eps(T) * state.v1_norm
        state.dimension = 1
        state.g2d = SVector{2,T}(state.v1_norm, zero(T))
        state.H2d = SMatrix{2,2,T}(state.v1_norm, zero(T), zero(T), one(T))
        return
    end
    
    if subspace.normalize
        @. cache.v2 /= v2_norm_ortho
        state.v2_norm = v2_norm_ortho
    end
    
    state.dimension = 2
    
    g1 = dot(cache.g, cache.v1)
    g2 = dot(cache.g, cache.v2)
    state.g2d = SVector{2,T}(g1, g2)

    apply_hessian!(cache.tmp, hess_approx, hess_state, cache, cache.v1)
    H11 = dot(cache.v1, cache.tmp)

    apply_hessian!(cache.tmp, hess_approx, hess_state, cache, cache.v2)
    H12 = dot(cache.v1, cache.tmp)

    H22 = dot(cache.v2, cache.tmp)
    
    state.H2d = SMatrix{2,2,T}(H11, H12, H12, H22)
end

"""
    solve_subspace_tr!(solver, subspace, state, cache, Δ) -> predicted_reduction

Solve the trust-region subproblem within the subspace.  Writes the step
into `cache.p` and returns the predicted reduction.
"""
function solve_subspace_tr!(solver, subspace::TwoDimSubspace, state, cache::RetroCache{T}, Δ::T) where {T}
    if state.dimension == 0
        fill!(cache.p, zero(T))
        return zero(T)
    elseif state.dimension == 1
        α = -state.g2d[1] / state.H2d[1,1]
        
        if subspace.normalize
            α = clamp(α, -Δ, Δ)
            @. cache.p = α * cache.v1
            return abs(α)
        else
            α = clamp(α, -Δ / max(state.v1_norm, eps(T)), Δ / max(state.v1_norm, eps(T)))
            @. cache.p = α * cache.v1
            return abs(α) * state.v1_norm
        end
    else

        solve_tr_2d!(solver, state.g2d, state.H2d, Δ, state)

        @. cache.p = state.p2d[1] * cache.v1 + state.p2d[2] * cache.v2
        return norm(state.p2d)
    end
end

function solve_tr_2d!(::EigenTRSolver, g2d::SVector{2,T}, H2d::SMatrix{2,2,T,4}, Δ::T, state) where {T}
    _solve_tr_2d_eigen!(g2d, H2d, Δ, state)
end

function solve_tr_2d!(::CauchyTRSolver, g2d::SVector{2,T}, H2d::SMatrix{2,2,T,4}, Δ::T, state) where {T}
    g_norm = norm(g2d)
    if g_norm < eps(T)
        state.p2d = @SVector zeros(T, 2)
        return
    end

    gHg = dot(g2d, H2d, g2d)
    if gHg > eps(T)
        α = min(g_norm^2 / gHg, Δ / g_norm)
    else
        α = Δ / g_norm
    end
    
    state.p2d = -α * g2d
end

function solve_tr_2d!(::AbstractTRSolver, g2d::SVector{2,T}, H2d::SMatrix{2,2,T,4}, Δ::T, state) where {T}
    _solve_tr_2d_eigen!(g2d, H2d, Δ, state)
end

function _solve_tr_2d_eigen!(g2d::SVector{2,T}, H2d::SMatrix{2,2,T,4}, Δ::T, state) where {T}
    g_norm = norm(g2d)
    if g_norm < eps(T)
        state.p2d = @SVector zeros(T, 2)
        return
    end
    
    det_H = H2d[1,1] * H2d[2,2] - H2d[1,2]^2
    
    if abs(det_H) > eps(T) * max(abs(H2d[1,1]), abs(H2d[2,2]))^2
        H_inv = SMatrix{2,2,T}(H2d[2,2], -H2d[1,2], -H2d[1,2], H2d[1,1]) / det_H
        p_newton = -H_inv * g2d
        newton_norm = norm(p_newton)
        
        if newton_norm <= Δ
            if H2d[1,1] > zero(T) && det_H > zero(T)
                state.p2d = p_newton
                return
            end
        end
    end

    trace_H = H2d[1,1] + H2d[2,2]
    discriminant = trace_H^2 - 4 * det_H
    
    if discriminant < zero(T)
        discriminant = zero(T)
    end
    
    sqrt_disc = sqrt(discriminant)
    λ1 = (trace_H - sqrt_disc) / 2 
    λ2 = (trace_H + sqrt_disc) / 2 

    if abs(H2d[1,2]) > eps(T)
        v1_raw = SVector{2,T}(H2d[1,2], λ1 - H2d[1,1])
        v1 = v1_raw / norm(v1_raw)
        v2 = SVector{2,T}(-v1[2], v1[1])
    elseif abs(H2d[1,1] - λ1) < abs(H2d[2,2] - λ1)
        v1 = SVector{2,T}(one(T), zero(T))
        v2 = SVector{2,T}(zero(T), one(T))
    else
        v1 = SVector{2,T}(zero(T), one(T))
        v2 = SVector{2,T}(one(T), zero(T))
    end

    g1 = dot(g2d, v1)
    g2 = dot(g2d, v2)
    
    λ_min = max(-λ1, zero(T)) + eps(T)
    λ_max = max(g_norm / Δ, abs(λ1), abs(λ2)) * 10  # Upper bound
    
    for _ in 1:20
        λ_mid = (λ_min + λ_max) / 2
        
        denom1 = λ1 + λ_mid
        denom2 = λ2 + λ_mid
        
        p1 = abs(denom1) > eps(T) ? g1 / denom1 : sign(g1) * Δ * 10
        p2 = abs(denom2) > eps(T) ? g2 / denom2 : sign(g2) * Δ * 10
        
        p_norm_sq = p1^2 + p2^2
        
        if p_norm_sq > Δ^2
            λ_min = λ_mid  
        else
            λ_max = λ_mid 
        end
    end

    λ_opt = (λ_min + λ_max) / 2
    p1 = abs(λ1 + λ_opt) > eps(T) ? -g1 / (λ1 + λ_opt) : zero(T)
    p2 = abs(λ2 + λ_opt) > eps(T) ? -g2 / (λ2 + λ_opt) : zero(T)

    state.p2d = p1 * v1 + p2 * v2

    p_norm = norm(state.p2d)
    if p_norm > eps(T) && p_norm < Δ * 0.99
        state.p2d = state.p2d * (Δ / p_norm)
    end
end