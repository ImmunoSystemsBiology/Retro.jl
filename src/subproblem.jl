# ============================================================================
# Subproblem Solvers
# ============================================================================

function solve_subproblem!(
    state::TrustRegionState{T, VT, MT},
    solver::AbstractSubproblemSolver,
) where {T, VT, MT}
    solver(state)
end

# ============================================================================
# 2D Subspace Solver
# ============================================================================

function (::TwoDimSubspace)(state::TrustRegionState{T}) where {T<:Real}
    g = gx(state)
    H = Hx(state)
    tr_radius = state.tr_radius
    n = length(g)
    
    # Use free gradient
    g_free = state.gx_free
    gnorm = norm(g_free)
    
    if gnorm < eps(T)
        state.step .= zero(T)
        state.last_step_norm = zero(T)
        return
    end
    
    # First direction: steepest descent (normalized)
    d1 = -g_free ./ gnorm
    
    # Compute Hg for second direction
    mul!(state.Hg, H, g_free)
    Hg_norm = norm(state.Hg)
    
    # Second direction: Newton-like (normalized)
    if Hg_norm > eps(T)
        d2 = -state.Hg ./ Hg_norm
    else
        # Fallback: use orthogonal direction
        d2 = similar(d1)
        d2 .= zero(T)
        if n >= 2
            d2[1] = -d1[2]
            d2[2] = d1[1]
        else
            d2[1] = one(T)
        end
        d2_norm = norm(d2)
        if d2_norm > eps(T)
            d2 ./= d2_norm
        end
    end
    
    # Orthogonalize d2 with respect to d1 (Gram-Schmidt)
    d2 .-= dot(d2, d1) .* d1
    d2_norm = norm(d2)
    if d2_norm > eps(T)
        d2 ./= d2_norm
    else
        # Directions are parallel, use only d1 (steepest descent)
        # Step length: min of Cauchy point or trust region radius
        gHg = dot(g_free, state.Hg)
        if gHg > eps(T)
            alpha_cauchy = gnorm^2 / gHg
        else
            alpha_cauchy = tr_radius
        end
        alpha = min(alpha_cauchy, tr_radius)
        
        state.step .= alpha .* d1
        state.step[state.active_set] .= zero(T)
        state.last_step_norm = alpha
        return
    end
    
    # Project gradient and Hessian onto 2D subspace
    g1 = dot(g_free, d1)
    g2 = dot(g_free, d2)
    
    # Compute Hessian in 2D subspace
    Hd1 = similar(d1)
    Hd2 = similar(d2)
    mul!(Hd1, H, d1)
    mul!(Hd2, H, d2)
    
    h11 = dot(d1, Hd1)
    h22 = dot(d2, Hd2)
    h12 = dot(d1, Hd2)
    
    # Solve 2D trust region subproblem
    alpha, beta = solve_2d_trust_region(g1, g2, h11, h22, h12, tr_radius)
    
    # Reconstruct step in full space
    state.step .= alpha .* d1 .+ beta .* d2
    
    # Cache step norm BEFORE zeroing out active variables
    state.last_step_norm = norm(state.step)
    
    # Zero out steps for active variables (shouldn't be needed if g_free is correct)
    state.step[state.active_set] .= zero(T)
end

# ============================================================================
# 2D Trust Region Solver (Exact)
# ============================================================================

function solve_2d_trust_region(g1::T, g2::T, h11::T, h22::T, h12::T, tr_radius::T) where T<:Real
    # Check for zero gradient
    g_norm_2d = sqrt(g1^2 + g2^2)
    if g_norm_2d < eps(T)
        return zero(T), zero(T)
    end
    
    # Try unconstrained Newton step first
    det_H = h11 * h22 - h12 * h12
    
    if abs(det_H) > eps(T) * max(abs(h11), abs(h22), one(T))
        # Solve 2x2 system: H * s = -g
        inv_det = one(T) / det_H
        s1 = -inv_det * (h22 * g1 - h12 * g2)
        s2 = -inv_det * (-h12 * g1 + h11 * g2)
        
        # Check if Newton step is within trust region
        s_norm_sq = s1^2 + s2^2
        if s_norm_sq ≤ tr_radius^2
            return s1, s2
        end
    end
    
    # Newton step outside trust region or H singular
    # Scale Newton step (or gradient if H singular) to trust region boundary
    
    # Try to compute Newton direction
    det_H = h11 * h22 - h12 * h12
    if abs(det_H) > eps(T) * max(abs(h11), abs(h22), one(T))
        inv_det = one(T) / det_H
        newton_s1 = -inv_det * (h22 * g1 - h12 * g2)
        newton_s2 = -inv_det * (-h12 * g1 + h11 * g2)
        newton_norm = sqrt(newton_s1^2 + newton_s2^2)
        
        if newton_norm > eps(T)
            # Scale to trust region boundary
            scale = tr_radius / newton_norm
            return scale * newton_s1, scale * newton_s2
        end
    end
    
    # Fallback: steepest descent to boundary
    if g_norm_2d > eps(T)
        scale = tr_radius / g_norm_2d
        return -scale * g1, -scale * g2
    else
        return zero(T), zero(T)
    end
end

# ============================================================================
# Steihaug-Toint CG Solver
# ============================================================================

function steihaug_cg!(state::TrustRegionState{T}, maxiter::Int) where T
    g = state.gx_free
    H = Hx(state)
    tr_radius = state.tr_radius
    n = length(g)
    
    # Initialize
    state.step .= zero(T)
    r = copy(g)  # Residual
    d = -copy(g)  # Search direction
    
    # Workspace
    Hd = similar(d)
    
    gnorm_sq = dot(g, g)
    if gnorm_sq < eps(T)
        state.last_step_norm = zero(T)
        return
    end
    
    for i in 1:min(maxiter, n)
        # Compute H * d
        mul!(Hd, H, d)
        dHd = dot(d, Hd)
        
        # Check for negative curvature
        if dHd ≤ eps(T)
            # Find tau such that ||step + tau*d|| = tr_radius
            tau = solve_quadratic_tr(state.step, d, tr_radius)
            @. state.step += tau * d
            state.step[state.active_set] .= zero(T)
            state.last_step_norm = norm(state.step)
            return
        end
        
        # Standard CG step size
        alpha = dot(r, r) / dHd
        
        # Check if step would leave trust region
        step_trial = state.step .+ alpha .* d
        if norm(step_trial) ≥ tr_radius
            # Find boundary point
            tau = solve_quadratic_tr(state.step, d, tr_radius)
            @. state.step += tau * d
            state.step[state.active_set] .= zero(T)
            state.last_step_norm = norm(state.step)
            return
        end
        
        # Store old residual norm squared for beta calculation
        r_old_norm_sq = dot(r, r)
        
        # Accept CG step
        @. state.step = step_trial
        @. r += alpha * Hd
        
        # Check residual convergence
        r_norm_sq = dot(r, r)
        if r_norm_sq < 1e-10 * gnorm_sq
            state.step[state.active_set] .= zero(T)
            state.last_step_norm = norm(state.step)
            return
        end
        
        # Compute next search direction using Polak-Ribière formula
        beta = r_norm_sq / r_old_norm_sq
        @. d = -r + beta * d
    end
    
    state.step[state.active_set] .= zero(T)
    state.last_step_norm = norm(state.step)
end

# ============================================================================
# Helper: Solve quadratic for trust region boundary
# ============================================================================

function solve_quadratic_tr(s::AbstractVector{T}, p::AbstractVector{T}, tr_radius::T) where T
    # Find tau >= 0 such that ||s + tau*p|| = tr_radius
    # This solves: ||s||^2 + 2*tau*(s'p) + tau^2*||p||^2 = tr_radius^2
    
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

# ============================================================================
# Full Space Solver (using eigenvalue method)
# ============================================================================

function full_space_solver!(state::TrustRegionState{T}) where T<:Real
    g = state.gx_free
    H = Hx(state)
    tr_radius = state.tr_radius
    
    # Try Cholesky first (if H is positive definite)
    try
        L = cholesky(H)
        state.step .= -(L \ g)
        
        step_norm = norm(state.step)
        if step_norm ≤ tr_radius
            # Newton step is within trust region
            state.step[state.active_set] .= zero(T)
            state.last_step_norm = step_norm
            return
        end
    catch
        # H not positive definite, continue to eigenvalue method
    end
    
    # Use eigenvalue regularization
    eigen_decomp = eigen(Symmetric(H))
    lambda_min = minimum(eigen_decomp.values)
    
    # Start with small regularization
    lambda = max(-lambda_min + 1e-8, 1e-8)
    
    for iter in 1:20
        H_reg = H + lambda * I
        try
            step_trial = -(H_reg \ g)
            step_norm = norm(step_trial)
            
            if abs(step_norm - tr_radius) < 1e-10 * tr_radius
                state.step .= step_trial
                state.step[state.active_set] .= zero(T)
                state.last_step_norm = step_norm
                return
            end
            
            if step_norm < tr_radius
                # Scale to boundary
                state.step .= step_trial .* (tr_radius / step_norm)
                state.step[state.active_set] .= zero(T)
                state.last_step_norm = tr_radius
                return
            else
                lambda *= 2
            end
        catch
            lambda *= 2
        end
    end
    
    # Fallback: steepest descent
    gnorm = norm(g)
    if gnorm > eps(T)
        state.step .= -(tr_radius / gnorm) .* g
    else
        state.step .= zero(T)
    end
    state.step[state.active_set] .= zero(T)
    state.last_step_norm = norm(state.step)
end

function (::FullSpace)(state::TrustRegionState)
    full_space_solver!(state)
end

function (solver::CGSubspace)(state::TrustRegionState)
    steihaug_cg!(state, solver.maxiter)
end