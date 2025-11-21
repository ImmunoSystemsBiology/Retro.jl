"""
    Trust-Region Subproblem Solvers

Methods for solving the trust-region subproblem:
    min_{s} m(s) = f + g'*s + 0.5*s'*H*s
    subject to ||s|| ≤ Δ

Different solvers offer tradeoffs between accuracy and computational cost.
"""

# ============================================================================
# Main dispatch function
# ============================================================================

"""
    solve_subproblem!(state, subspace)

Solve the trust-region subproblem and store the step in `state.step`.

Dispatches to the appropriate solver based on the subspace type.
"""
function solve_subproblem!(
    state::TrustRegionState{T, VT, MT},
    subspace::AbstractSubspace,
) where {T, VT, MT}
    # Compute Coleman-Li affine scaling
    compute_affine_scaling!(state)
    
    # Get free variables (not at active bounds)
    g_free = state.gx_free
    n_free = length(g_free)
    
    if n_free == 0
        state.step .= zero(T)
        state.last_step_norm = zero(T)
        return
    end
    
    # Apply Coleman-Li scaling transformation
    # Scaled gradient: sg = D*g where D = Diagonal(sqrt(|v|))
    sg_free = similar(g_free)
    free_indices = findall(.!state.active_set)
    
    @inbounds for (i, idx) in enumerate(free_indices)
        sg_free[i] = sqrt(abs(state.v[idx])) * g_free[i]
    end
    
    # Scaled Hessian: SH = D*H*D + diag(g .* dv) where D = Diagonal(sqrt(|v|))
    # For free variables only
    D_free = Diagonal([sqrt(abs(state.v[idx])) for idx in free_indices])
    H_free = state.hessian[.!state.active_set, .!state.active_set]
    
    # Compute scaled Hessian: D*H*D
    shess = D_free * H_free * D_free
    
    # Add diagonal correction: diag(g .* dv)
    g_dv = [state.grad[idx] * state.dv[idx] for idx in free_indices]
    @inbounds for i in 1:n_free
        shess[i, i] += g_dv[i]
    end
    
    # Dispatch to specific subspace solver (works in scaled space)
    solve_subproblem_dispatch!(state, subspace, sg_free, shess, D_free, free_indices)
end

# ============================================================================
# TwoDimSubspace solver
# ============================================================================

function solve_subproblem_dispatch!(
    state::TrustRegionState{T, VT, MT},
    subspace::TwoDimSubspace,
    sg_free::AbstractVector{T},
    shess::AbstractMatrix{T},
    D_free::Diagonal{T, Vector{T}},
    free_indices::Vector{Int}
) where {T, VT, MT}
    
    gnorm = norm(sg_free)
    tol = sqrt(eps(T))
    
    if gnorm <= tol
        state.step .= zero(T)
        state.last_step_norm = zero(T)
        return
    end
    
    # First direction: steepest descent (normalized)
    d1 = -sg_free ./ gnorm
    
    # Compute shess * sg_free for second direction
    Hg = shess * sg_free
    
    # Check if Hessian is positive definite for Newton direction
    d2 = -Hg
    d2_norm = norm(d2)
    
    if d2_norm <= tol
        # If Hessian gives no useful direction, use orthogonal fallback
        d2 = construct_orthogonal_direction(d1, length(d1))
        d2_norm = norm(d2)
    end
    
    # Gram-Schmidt orthogonalization
    d2 .-= dot(d2, d1) .* d1
    d2_norm = norm(d2)
    
    if d2_norm <= tol
        # Directions are parallel - use 1D subproblem (Cauchy point)
        gHg = dot(sg_free, Hg)
        
        # Cauchy point in scaled space
        # Since d1 = -sg/gnorm is a unit vector, and ss = alpha * d1 = -(alpha/gnorm)*sg
        # To minimize m(ss) = sg'*ss + 0.5*ss'*H*ss along -sg direction:
        # Setting ss = -(alpha/gnorm)*sg and minimizing gives alpha = gnorm^3 / gHg
        if gHg > eps(T)
            alpha_cauchy = gnorm^3 / gHg
            alpha = min(alpha_cauchy, state.tr_radius / gnorm)
        else
            # Hessian is indefinite - use steepest descent to boundary
            alpha = state.tr_radius / gnorm
        end
        
        # Step in scaled space
        ss = alpha .* d1
        
        # Compute predicted reduction in scaled space: -m(ss) = -(sg'*ss + 0.5*ss'*shess*ss)
        Hss = shess * ss
        sg_dot_ss = dot(sg_free, ss)
        ss_H_ss = dot(ss, Hss)
        state.predicted_reduction = -(sg_dot_ss + T(0.5) * ss_H_ss)
        
        # Transform back to original space
        state.step .= zero(T)
        state.step[free_indices] .= D_free * ss
        state.step[state.active_set] .= zero(T)
        state.last_step_norm = norm(state.step)
        return
    end
    
    # Normalize d2
    d2 ./= d2_norm
    
    # Project gradient and Hessian onto 2D subspace
    g1 = dot(sg_free, d1)
    g2 = dot(sg_free, d2)
    
    # Compute Hessian in 2D subspace
    Hd1 = shess * d1
    Hd2 = shess * d2
    
    h11 = dot(d1, Hd1)
    h22 = dot(d2, Hd2)
    h12 = dot(d1, Hd2)
    
    # Solve 2D trust region subproblem
    alpha, beta = solve_2d_subproblem(subspace.solver, g1, g2, h11, h22, h12, state.tr_radius)
    
    # Reconstruct step in scaled space
    ss = alpha .* d1 .+ beta .* d2
    
    # Compute predicted reduction in scaled space: -m(ss) = -(sg'*ss + 0.5*ss'*shess*ss)
    Hss = shess * ss
    sg_dot_ss = dot(sg_free, ss)
    ss_H_ss = dot(ss, Hss)
    state.predicted_reduction = -(sg_dot_ss + T(0.5) * ss_H_ss)
    
    # Transform back to original space: s = D * ss
    state.step .= zero(T)
    state.step[free_indices] .= D_free * ss
    state.step[state.active_set] .= zero(T)
    state.last_step_norm = norm(state.step)
end

"""
    construct_orthogonal_direction(v, n)

Construct a vector orthogonal to v using index swapping trick.
"""
function construct_orthogonal_direction(v::AbstractVector{T}, n::Int) where T
    d = zeros(T, n)
    if n == 1
        d[1] = one(T)
    else
        j = argmax(abs.(v))
        k = j == 1 ? 2 : 1
        d[k] = -v[j]
        d[j] = v[k]
    end
    return d
end

# ============================================================================
# 2D Subproblem Solvers
# ============================================================================

"""
    solve_2d_subproblem(solver, g1, g2, h11, h22, h12, tr_radius)

Solve 2D trust-region subproblem with the specified solver.

Returns (alpha, beta) coefficients in the 2D subspace.
"""
function solve_2d_subproblem(
    solver::EigenvalueSolver,
    g1::T, g2::T, h11::T, h22::T, h12::T, tr_radius::T
) where T<:Real
    
    # Compute eigenvalues of 2x2 Hessian
    trace_H = h11 + h22
    det_H = h11 * h22 - h12^2
    discriminant = trace_H^2 - 4*det_H
    
    if discriminant < 0
        # Complex eigenvalues shouldn't happen for symmetric matrix
        # Fall back to Cauchy point
        return solve_2d_cauchy_point(g1, g2, h11, h22, h12, tr_radius)
    end
    
    sqrt_disc = sqrt(max(discriminant, zero(T)))
    λ1 = (trace_H + sqrt_disc) / 2  # Larger eigenvalue
    λ2 = (trace_H - sqrt_disc) / 2  # Smaller eigenvalue
    
    # Check if positive definite
    if λ2 > zero(T)
        # Try unconstrained Newton step
        if abs(det_H) > eps(T) * max(abs(h11), abs(h22), one(T))
            inv_det = one(T) / det_H
            s1 = -inv_det * (h22 * g1 - h12 * g2)
            s2 = -inv_det * (-h12 * g1 + h11 * g2)
            s_norm = sqrt(s1^2 + s2^2)
            
            if s_norm <= tr_radius + sqrt(eps(T))
                # Interior solution
                return s1, s2
            end
        end
        λ_init = zero(T)
    else
        λ_init = -λ2 + sqrt(eps(T))
    end
    
    # Need boundary solution via secular equation
    # Try Newton's method first
    f(λ) = secular_2d(λ, g1, g2, h11, h22, h12, tr_radius)
    df(λ) = dsecular_2d(λ, g1, g2, h11, h22, h12, tr_radius)
    
    λ_sol = λ_init
    converged = false
    
    for iter in 1:solver.max_newton_iterations
        fval = f(λ_sol)
        if abs(fval) < solver.newton_tolerance
            converged = true
            break
        end
        
        dfval = df(λ_sol)
        if abs(dfval) < eps(T)
            break
        end
        
        λ_sol -= fval / dfval
        λ_sol = max(λ_sol, λ_init)  # Keep λ >= λ_min
    end
    
    # If Newton failed, use ITP bracketing solver
    if !converged && f(λ_init) < -solver.newton_tolerance
        # Find upper bound
        λ_low = λ_init
        λ_high = λ_init + one(T)
        max_iter = solver.max_newton_iterations
        
        while f(λ_high) < zero(T) && max_iter > 0
            λ_low = λ_high
            λ_high *= 10
            max_iter -= 1
        end
        
        if max_iter > 0 && f(λ_high) > zero(T)
            # Use ITP solver
            f_itp(u, p) = f(u)
            prob_itp = NLS.IntervalNonlinearProblem(f_itp, (λ_low, λ_high))
            sol_itp = NLS.solve(prob_itp, NLS.ITP(); maxiters = solver.max_newton_iterations)
            λ_sol = sol_itp.u
        end
    end
    
    # Compute step at λ_sol
    s1, s2 = s_lambda_2d(λ_sol, g1, g2, h11, h22, h12)
    
    # Check if solution is valid
    if sqrt(s1^2 + s2^2) <= tr_radius + sqrt(eps(T))
        return s1, s2
    end
    
    # HARD CASE: gradient orthogonal to eigenvector of smallest eigenvalue
    # Step along eigenvector direction
    if abs(h12) > eps(T)
        eigvec1 = one(T) / sqrt(one(T) + ((λ2 - h11) / h12)^2)
        eigvec2 = -((λ2 - h11) / h12) * eigvec1
    else
        eigvec1 = one(T)
        eigvec2 = zero(T)
    end
    
    # Normalize
    eigvec_norm = sqrt(eigvec1^2 + eigvec2^2)
    eigvec1 /= eigvec_norm
    eigvec2 /= eigvec_norm
    
    s1 = tr_radius * eigvec1
    s2 = tr_radius * eigvec2
    
    return s1, s2
end

function solve_2d_subproblem(
    solver::CauchyPointSolver,
    g1::T, g2::T, h11::T, h22::T, h12::T, tr_radius::T
) where T<:Real
    return solve_2d_cauchy_point(g1, g2, h11, h22, h12, tr_radius)
end

"""
    solve_2d_cauchy_point(g1, g2, h11, h22, h12, tr_radius)

Compute Cauchy point in 2D subspace (fast approximate solution).
"""
function solve_2d_cauchy_point(
    g1::T, g2::T, h11::T, h22::T, h12::T, tr_radius::T
) where T<:Real
    
    g_norm_2d = sqrt(g1^2 + g2^2)
    if g_norm_2d < eps(T)
        return zero(T), zero(T)
    end
    
    # Compute gHg in 2D
    gHg = g1*g1*h11 + 2*g1*g2*h12 + g2*g2*h22
    
    if gHg > eps(T)
        # Positive curvature
        t_cauchy = g_norm_2d^2 / gHg
        t = min(t_cauchy, tr_radius / g_norm_2d)
    else
        # Non-positive curvature - go to boundary
        t = tr_radius / g_norm_2d
    end
    
    return -t * g1, -t * g2
end

# ============================================================================
# Helper functions for 2D subproblem
# ============================================================================

function s_lambda_2d(λ::T, g1::T, g2::T, h11::T, h22::T, h12::T) where T<:Real
    h11_λ = h11 + λ
    h22_λ = h22 + λ
    det_λ = h11_λ * h22_λ - h12^2
    
    if abs(det_λ) < sqrt(eps(T))
        return zero(T), zero(T)
    end
    
    inv_det = one(T) / det_λ
    s1 = -inv_det * (h22_λ * g1 - h12 * g2)
    s2 = -inv_det * (-h12 * g1 + h11_λ * g2)
    
    return s1, s2
end

function secular_2d(λ::T, g1::T, g2::T, h11::T, h22::T, h12::T, tr_radius::T) where T<:Real
    s1, s2 = s_lambda_2d(λ, g1, g2, h11, h22, h12)
    s_norm = sqrt(s1^2 + s2^2)
    
    if s_norm > zero(T)
        return 1/s_norm - 1/tr_radius
    else
        return T(Inf)
    end
end

function dsecular_2d(λ::T, g1::T, g2::T, h11::T, h22::T, h12::T, tr_radius::T) where T<:Real
    h11_λ = h11 + λ
    h22_λ = h22 + λ
    det_λ = h11_λ * h22_λ - h12^2
    
    if abs(det_λ) < sqrt(eps(T))
        return T(Inf)
    end
    
    s1, s2 = s_lambda_2d(λ, g1, g2, h11, h22, h12)
    s_norm = sqrt(s1^2 + s2^2)
    
    if s_norm < sqrt(eps(T))
        return T(Inf)
    end
    
    # Derivative of s with respect to λ: ds/dλ = -(H + λI)^{-2} * s
    # Since s = -(H + λI)^{-1} * g, we have ds/dλ = (H + λI)^{-2} * g
    # But more accurately: d/dλ[(H + λI)^{-1}*g] = -(H + λI)^{-2}*g
    inv_det = one(T) / det_λ
    
    # (H + λI)^{-1} has entries that are polynomial in λ/det_λ
    # Derivative: d(inv_det)/dλ = -inv_det^2 * d(det_λ)/dλ
    # where d(det_λ)/dλ = trace(H + λI)
    d_det_λ = h11_λ + h22_λ
    d_inv_det = -inv_det * inv_det * d_det_λ
    
    # ds/dλ components using product rule
    ds1_dλ = d_inv_det * (h22_λ * g1 - h12 * g2) + inv_det * g1
    ds2_dλ = d_inv_det * (-h12 * g1 + h11_λ * g2) + inv_det * g2
    
    # d||s||/dλ using chain rule
    ds_norm_dλ = (s1 * ds1_dλ + s2 * ds2_dλ) / s_norm
    
    # d(1/||s||)/dλ = -1/||s||^2 * d||s||/dλ
    return -ds_norm_dλ / (s_norm^2)
end

# ============================================================================
# FullSpace (N-dimensional) solver
# ============================================================================

function solve_subproblem_dispatch!(
    state::TrustRegionState{T, VT, MT},
    subspace::FullSpace,
    sg_free::AbstractVector{T},
    shess::AbstractMatrix{T},
    D_free::Diagonal{T, Vector{T}},
    free_indices::Vector{Int}
) where {T, VT, MT}
    
    gnorm = norm(sg_free)
    if gnorm <= sqrt(eps(T))
        state.step .= zero(T)
        state.last_step_norm = zero(T)
        return
    end
    
    # Solve in scaled space
    ss, step_type = solve_nd_subproblem(subspace.solver, shess, sg_free, state.tr_radius)
    
    # Compute predicted reduction in scaled space: -m(ss) = -(sg'*ss + 0.5*ss'*shess*ss)
    Hss = shess * ss
    state.predicted_reduction = -(dot(sg_free, ss) + T(0.5) * dot(ss, Hss))
    
    # Transform back to original space: s = D * ss
    state.step .= zero(T)
    state.step[free_indices] .= D_free * ss
    state.step[state.active_set] .= zero(T)
    state.last_step_norm = norm(state.step)
end

"""
    solve_nd_subproblem(solver, B, g, delta)

Solve n-dimensional trust-region subproblem using eigenvalue decomposition.

This is the exact solver for the general trust-region subproblem in n dimensions.
Returns (s, step_type) where step_type indicates the solution type.
"""
function solve_nd_subproblem(
    solver::EigenvalueSolver,
    B::AbstractMatrix{T},
    g::AbstractVector{T},
    delta::T
) where T<:Real
    
    if delta == 0
        return zeros(T, length(g)), "zero"
    end
    
    # Eigenvalue decomposition
    eigvals, eigvecs = eigen(Symmetric(B))
    eigvals = real.(eigvals)
    eigvecs = real.(eigvecs)
    
    w = -eigvecs' * g
    jmin = argmin(eigvals)
    mineig = eigvals[jmin]
    
    # POSITIVE DEFINITE CASE
    if mineig > zero(T)
        s = slam_nd(zero(T), w, eigvals, eigvecs)
        if norm(s) <= delta + sqrt(eps(T))
            return s, "posdef"
        end
        λ_init = zero(T)
    else
        λ_init = -mineig + sqrt(eps(T))
    end
    
    # INDEFINITE CASE - need to solve secular equation
    f(λ) = secular_nd(λ, w, eigvals, eigvecs, delta)
    df(λ) = dsecular_nd(λ, w, eigvals, eigvecs, delta)
    
    if f(λ_init) < -solver.newton_tolerance
        # Try Newton's method
        λ_sol = λ_init
        converged = false
        
        for iter in 1:solver.max_newton_iterations
            fval = f(λ_sol)
            if abs(fval) < solver.newton_tolerance
                converged = true
                break
            end
            
            dfval = df(λ_sol)
            if abs(dfval) < eps(T)
                break
            end
            
            λ_sol -= fval / dfval
            λ_sol = max(λ_sol, λ_init)
        end
        
        # Try ITP if Newton failed
        if !converged
            λ_low = λ_init
            λ_high = λ_init + one(T)
            max_iter = solver.max_newton_iterations
            
            while f(λ_high) < zero(T) && max_iter > 0
                λ_low = λ_high
                λ_high *= 10
                max_iter -= 1
            end
            
            if max_iter > 0 && f(λ_high) > zero(T)
                f_itp(u, p) = f(u)
                prob_itp = NLS.IntervalNonlinearProblem(f_itp, (λ_low, λ_high))
                sol_itp = NLS.solve(prob_itp, NLS.ITP(); maxiters = solver.max_newton_iterations)
                λ_sol = sol_itp.u
            end
        end
        
        s = slam_nd(λ_sol, w, eigvals, eigvecs)
        if norm(s) <= delta + sqrt(eps(T))
            return s, "indef"
        end
    end
    
    # HARD CASE
    w_hard = copy(w)
    w_hard[abs.(eigvals .- mineig) .< sqrt(eps(T))] .= zero(T)
    s = slam_nd(-mineig, w_hard, eigvals, eigvecs)
    
    # Compute sigma to reach boundary
    sigma = sqrt(max(delta^2 - norm(s)^2, zero(T)))
    s .+= sigma .* eigvecs[:, jmin]
    
    return s, "hard"
end

# ============================================================================
# Helper functions for N-D subproblem
# ============================================================================

function slam_nd(
    λ::T,
    w::AbstractVector{T},
    eigvals::AbstractVector{T},
    eigvecs::AbstractMatrix{T}
) where T<:Real
    c = copy(w)
    el = eigvals .+ λ
    mask = abs.(el) .> eps(T)
    c[mask] ./= el[mask]
    return eigvecs * c
end

function dslam_nd(
    λ::T,
    w::AbstractVector{T},
    eigvals::AbstractVector{T},
    eigvecs::AbstractMatrix{T}
) where T<:Real
    c = copy(w)
    el = eigvals .+ λ
    
    mask_nonzero = abs.(el) .> eps(T)
    c[mask_nonzero] ./= -(el[mask_nonzero].^2)
    
    mask_singular = (.!mask_nonzero) .& (abs.(c) .> eps(T))
    c[mask_singular] .= T(Inf)
    
    return eigvecs * c
end

function secular_nd(
    λ::T,
    w::AbstractVector{T},
    eigvals::AbstractVector{T},
    eigvecs::AbstractMatrix{T},
    delta::T
) where T<:Real
    if λ < -minimum(eigvals)
        return T(Inf)
    end
    
    s = slam_nd(λ, w, eigvals, eigvecs)
    sn = norm(s)
    
    if sn > zero(T)
        return 1/sn - 1/delta
    else
        return T(Inf)
    end
end

function dsecular_nd(
    λ::T,
    w::AbstractVector{T},
    eigvals::AbstractVector{T},
    eigvecs::AbstractMatrix{T},
    delta::T
) where T<:Real
    s = slam_nd(λ, w, eigvals, eigvecs)
    ds = dslam_nd(λ, w, eigvals, eigvecs)
    sn = norm(s)
    
    if sn > zero(T)
        return -dot(s, ds) / (sn^3)
    else
        return T(Inf)
    end
end

# ============================================================================
# CGSubspace (Steihaug-Toint) solver
# ============================================================================

function solve_subproblem_dispatch!(
    state::TrustRegionState{T, VT, MT},
    subspace::CGSubspace,
    sg_free::AbstractVector{T},
    shess::AbstractMatrix{T},
    D_free::Diagonal{T, Vector{T}},
    free_indices::Vector{Int}
) where {T, VT, MT}
    
    gnorm = norm(sg_free)
    if gnorm <= sqrt(eps(T))
        state.step .= zero(T)
        state.last_step_norm = zero(T)
        state.predicted_reduction = zero(T)
        return
    end
    
    # Compute predicted reduction (need to implement in steihaug_toint!)
    ss = steihaug_toint!(state, sg_free, shess, state.tr_radius, subspace.maxiter)
    
    # Transform back to original space: s = D * ss
    state.step .= zero(T)
    @inbounds for (i, idx) in enumerate(free_indices)
        state.step[idx] = D_free.diag[i] * ss[i]
    end
    state.last_step_norm = norm(state.step)
end

"""
    steihaug_toint!(g, H, delta, maxiter)

Steihaug-Toint truncated conjugate gradient method for trust-region subproblem.

This method uses conjugate gradients to approximately solve the trust-region
subproblem. It terminates early if:
1. Negative curvature is encountered
2. The trust-region boundary is hit
3. The CG residual is sufficiently small

This is efficient for large-scale problems where forming/factoring the full
Hessian is expensive.
"""
function steihaug_toint!(
    state::TrustRegionState{T, VT, MT},
    g::AbstractVector{T},
    H::AbstractMatrix{T},
    delta::T,
    maxiter::Int
) where {T<:Real, VT, MT}
    
    n = length(g)
    z = zeros(T, n)
    r = copy(g)
    d = -copy(g)
    
    gnorm_sq = dot(g, g)
    eps_cg = min(T(0.5), sqrt(gnorm_sq)) * gnorm_sq
    
    if gnorm_sq < eps_cg
        state.predicted_reduction = zero(T)
        return z
    end
    
    for iter in 1:min(maxiter, n)
        Hd = H * d
        dHd = dot(d, Hd)
        
        # Negative curvature - go to boundary along d
        if dHd <= eps(T)
            tau = solve_quadratic_tr(z, d, delta)
            ss = z .+ tau .* d
            # Compute predicted reduction: -m(ss) = -(g'*ss + 0.5*ss'*H*ss)
            Hss = H * ss
            state.predicted_reduction = -(dot(g, ss) + T(0.5) * dot(ss, Hss))
            return ss
        end
        
        # Standard CG step
        alpha = dot(r, r) / dHd
        z_new = z .+ alpha .* d
        
        # Check if step would leave trust region
        if norm(z_new) >= delta
            tau = solve_quadratic_tr(z, d, delta)
            ss = z .+ tau .* d
            # Compute predicted reduction: -m(ss) = -(g'*ss + 0.5*ss'*H*ss)
            Hss = H * ss
            state.predicted_reduction = -(dot(g, ss) + T(0.5) * dot(ss, Hss))
            return ss
        end
        
        # Update CG iteration
        r_old_norm_sq = dot(r, r)
        r .+= alpha .* Hd
        r_norm_sq = dot(r, r)
        
        # Check convergence
        if r_norm_sq < eps_cg
            # Compute predicted reduction: -m(ss) = -(g'*z_new + 0.5*z_new'*H*z_new)
            Hz = H * z_new
            state.predicted_reduction = -(dot(g, z_new) + T(0.5) * dot(z_new, Hz))
            return z_new
        end
        
        beta = r_norm_sq / r_old_norm_sq
        d .= -r .+ beta .* d
        z .= z_new
    end
    
    # Maxiter reached - compute predicted reduction for final z
    Hz = H * z
    state.predicted_reduction = -(dot(g, z) + T(0.5) * dot(z, Hz))
    return z
end

"""
    solve_quadratic_tr(s, p, tr_radius)

Find tau >= 0 such that ||s + tau*p|| = tr_radius.

Solves the quadratic equation: ||s||^2 + 2*tau*(s'p) + tau^2*||p||^2 = tr_radius^2
"""
function solve_quadratic_tr(
    s::AbstractVector{T},
    p::AbstractVector{T},
    tr_radius::T
) where T<:Real
    
    sp = dot(s, p)
    pp = dot(p, p)
    ss = dot(s, s)
    
    if pp < eps(T)
        return zero(T)
    end
    
    # Quadratic formula: a*tau^2 + b*tau + c = 0
    a = pp
    b = 2 * sp
    c = ss - tr_radius^2
    
    discriminant = b^2 - 4*a*c
    if discriminant < 0
        return zero(T)
    end
    
    sqrt_disc = sqrt(discriminant)
    tau1 = (-b - sqrt_disc) / (2*a)
    tau2 = (-b + sqrt_disc) / (2*a)
    
    # Return positive root
    if tau1 > 0
        return tau1
    else
        return max(tau2, zero(T))
    end
end
