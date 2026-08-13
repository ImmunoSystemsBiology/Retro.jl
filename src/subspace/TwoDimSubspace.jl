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
function build_subspace!(subspace::TwoDimSubspace, state, cache::RetroCache{T}, hess_approx, hess_state, x, Δ::T) where {T}
    g = cache.g
    g_norm = norm(g)

    if g_norm < eps(T)
        state.dimension = 0
        state.g2d = @SVector zeros(T, 2)
        state.H2d = @SMatrix zeros(T, 2, 2)
        return
    end

    if hess_approx isa SR1
        copy!(cache.v1, g)
        v1_norm = norm(cache.v1)
        if v1_norm < eps(T)
            state.dimension = 0
            state.g2d = @SVector zeros(T, 2)
            state.H2d = @SMatrix zeros(T, 2, 2)
            return
        end
        @. cache.v1 /= v1_norm

        solve_newton_direction!(cache.v2, hess_approx, hess_state, cache, g)
        v1_dot_v2 = dot(cache.v1, cache.v2)
        @. cache.v2 -= v1_dot_v2 * cache.v1
        v2_norm = norm(cache.v2)

        if v2_norm < eps(T) * v1_norm
            state.dimension = 1
            state.v1_norm = one(T)
            state.v2_norm = zero(T)
            state.g2d = SVector{2,T}(v1_norm, zero(T))
            H11 = dot(cache.v1, cache.B, cache.v1)
            state.H2d = SMatrix{2,2,T}(H11, zero(T), zero(T), one(T))
            return
        end

        @. cache.v2 /= v2_norm
        state.dimension = 2
        state.v1_norm = one(T)
        state.v2_norm = one(T)

        g1 = dot(g, cache.v1)
        g2 = dot(g, cache.v2)
        apply_hessian!(cache.tmp, hess_approx, hess_state, cache, cache.v1)
        H11 = dot(cache.v1, cache.tmp)
        apply_hessian!(cache.tmp, hess_approx, hess_state, cache, cache.v2)
        H12 = dot(cache.v1, cache.tmp)
        H22 = dot(cache.v2, cache.tmp)
        state.g2d = SVector{2,T}(g1, g2)
        state.H2d = SMatrix{2,2,T}(H11, H12, H12, H22)
        return
    end

    B = cache.B
    posdef = false
    try
        cholesky(Symmetric(B); check = true)
        posdef = true
    catch
        posdef = false
    end

    try
        F = cholesky(Symmetric(B), check = false)
        if issuccess(F)
            cache.v1 .= -(F \ g)
        else
            cache.v1 .= -(B \ g)
        end
    catch
        @. cache.v1 = -g
    end
    s_newt_norm = norm(cache.v1)

    if posdef
        if s_newt_norm < Δ
            if s_newt_norm < eps(T)
                state.dimension = 0
                state.g2d = @SVector zeros(T, 2)
                state.H2d = @SMatrix zeros(T, 2, 2)
                return
            end

            @. cache.v1 /= s_newt_norm
            state.dimension = 1
            state.v1_norm = one(T)
            state.v2_norm = zero(T)
            g1 = dot(g, cache.v1)
            H11 = dot(cache.v1, B, cache.v1)
            state.g2d = SVector{2,T}(g1, zero(T))
            state.H2d = SMatrix{2,2,T}(H11, zero(T), zero(T), one(T))
            return
        end

        copy!(cache.v2, g)
    else
        eig = eigen(Symmetric(B))
        imin = argmin(eig.values)
        @views cache.v1 .= eig.vectors[:, imin]

        has_scaling = true
        @inbounds for i in eachindex(cache.scaling)
            if cache.scaling[i] <= zero(T)
                has_scaling = false
                break
            end
        end

        if has_scaling
            @inbounds for i in eachindex(g)
                sgn = g[i] == zero(T) ? one(T) : sign(g[i])
                cache.v2[i] = cache.scaling[i] * sgn
            end
        else
            copy!(cache.v2, g)
        end
    end

    v1_norm = norm(cache.v1)
    if v1_norm < eps(T)
        state.dimension = 0
        state.g2d = @SVector zeros(T, 2)
        state.H2d = @SMatrix zeros(T, 2, 2)
        return
    end
    @. cache.v1 /= v1_norm

    proj = dot(cache.v2, cache.v1)
    @. cache.v2 = cache.v2 - proj * cache.v1
    v2_norm = norm(cache.v2)

    if v2_norm <= sqrt(eps(T))
        state.dimension = 1
        state.v1_norm = one(T)
        state.v2_norm = zero(T)
        g1 = dot(g, cache.v1)
        H11 = dot(cache.v1, B, cache.v1)
        state.g2d = SVector{2,T}(g1, zero(T))
        state.H2d = SMatrix{2,2,T}(H11, zero(T), zero(T), one(T))
        return
    end

    @. cache.v2 /= v2_norm
    state.dimension = 2
    state.v1_norm = one(T)
    state.v2_norm = one(T)

    g1 = dot(g, cache.v1)
    g2 = dot(g, cache.v2)
    H11 = dot(cache.v1, B, cache.v1)
    H12 = dot(cache.v1, B, cache.v2)
    H22 = dot(cache.v2, B, cache.v2)

    state.g2d = SVector{2,T}(g1, g2)
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
        denom = state.H2d[1,1]
        if abs(denom) > eps(T)
            α = -state.g2d[1] / denom
        else
            α = -sign(state.g2d[1]) * Δ
        end
        
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
    g1 = g2d[1]
    g2 = g2d[2]
    a = H2d[1, 1]
    b = H2d[1, 2]
    c = H2d[2, 2]

    if abs(g1) <= eps(T) && abs(g2) <= eps(T)
        state.p2d = zero(SVector{2, T})
        return zero(T)
    end

    # Closed-form eigendecomposition for a symmetric 2×2 matrix.
    trace = a + c
    delta = a - c
    disc = hypot(delta, T(2) * b)
    λ1 = T(0.5) * (trace + disc)
    λ2 = T(0.5) * (trace - disc)

    if abs(b) <= eps(T)
        v1x, v1y = one(T), zero(T)
        v2x, v2y = zero(T), one(T)
        if a < c
            v1x, v1y = zero(T), one(T)
            v2x, v2y = one(T), zero(T)
        end
    else
        v1x = b
        v1y = λ1 - a
        n1 = hypot(v1x, v1y)
        if n1 <= eps(T)
            v1x, v1y = one(T), zero(T)
            n1 = one(T)
        else
            v1x /= n1
            v1y /= n1
        end

        v2x = b
        v2y = λ2 - a
        n2 = hypot(v2x, v2y)
        if n2 <= eps(T)
            v2x, v2y = zero(T), one(T)
            n2 = one(T)
        else
            v2x /= n2
            v2y /= n2
        end
    end

    g1e = v1x * g1 + v1y * g2
    g2e = v2x * g1 + v2y * g2

    # Fast unconstrained check: if all eigenvalues are positive and the step is already inside the trust region,
    # no secular solve is needed.
    if λ1 > eps(T) && λ2 > eps(T)
        p1 = -g1e / λ1
        p2 = -g2e / λ2
        p_norm_sq = p1 * p1 + p2 * p2
        if p_norm_sq <= Δ * Δ
            p1v = v1x * p1 + v2x * p2
            p2v = v1y * p1 + v2y * p2
            state.p2d = SVector{2, T}(p1v, p2v)
            return sqrt(p_norm_sq)
        end
    end

    λ_min = min(λ1, λ2)
    σ_lo = max(zero(T), -λ_min + T(1e-8))
    σ_hi = max(σ_lo + one(T), one(T))

    function secular_norm_sq(σ::T)
        denom1 = λ1 + σ
        denom2 = λ2 + σ
        return (g1e / denom1)^2 + (g2e / denom2)^2
    end

    while sqrt(secular_norm_sq(σ_hi)) >= Δ
        σ_hi *= T(2)
        if σ_hi > T(1e12)
            break
        end
    end

    for _ in 1:80
        σ_mid = (σ_lo + σ_hi) / T(2)
        f_mid = sqrt(secular_norm_sq(σ_mid)) - Δ
        if abs(f_mid) <= T(1e-12) * max(Δ, one(T))
            σ_hi = σ_mid
            break
        end
        if f_mid > zero(T)
            σ_lo = σ_mid
        else
            σ_hi = σ_mid
        end
    end

    σ = (σ_lo + σ_hi) / T(2)
    q1 = -g1e / (λ1 + σ)
    q2 = -g2e / (λ2 + σ)
    p1v = v1x * q1 + v2x * q2
    p2v = v1y * q1 + v2y * q2
    state.p2d = SVector{2, T}(p1v, p2v)
    return hypot(p1v, p2v)
end

function solve_tr_2d!(::CauchyTRSolver, g2d::SVector{2,T}, H2d::SMatrix{2,2,T,4}, Δ::T, state) where {T}
    g_norm = hypot(g2d[1], g2d[2])
    if g_norm <= eps(T)
        state.p2d = zero(SVector{2, T})
        return zero(T)
    end

    a = H2d[1, 1]
    b = H2d[1, 2]
    c = H2d[2, 2]
    g1 = g2d[1]
    g2 = g2d[2]
    gHg = a * g1 * g1 + T(2) * b * g1 * g2 + c * g2 * g2

    if gHg > eps(T)
        α = min(g_norm^2 / gHg, Δ / g_norm)
    else
        α = Δ / g_norm
    end

    p1 = -α * g1
    p2 = -α * g2
    state.p2d = SVector{2, T}(p1, p2)
    return α * g_norm
end

function solve_tr_2d!(::AbstractTRSolver, g2d::SVector{2,T}, H2d::SMatrix{2,2,T,4}, Δ::T, state) where {T}
    solve_tr_2d!(EigenTRSolver{T}(), g2d, H2d, Δ, state)
end