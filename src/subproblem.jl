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

function (solver::TwoDimSubspace{F})(state::TrustRegionState{T}) where {F<:AbstractSubproblemFallback, T<:Real}
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
    
    # Solve 2D trust region subproblem (dispatch on fallback type)
    alpha, beta = solve_2d_trust_region(solver.fallback, g1, g2, h11, h22, h12, tr_radius)
    
    # Reconstruct step in full space
    state.step .= alpha .* d1 .+ beta .* d2
    
    # Cache step norm BEFORE zeroing out active variables
    state.last_step_norm = norm(state.step)
    
    # Zero out steps for active variables (shouldn't be needed if g_free is correct)
    state.step[state.active_set] .= zero(T)
end

# ============================================================================
# 2D Trust Region Solver Dispatch (by fallback type)
# ============================================================================

# Cauchy point fallback (default, fast)
function solve_2d_trust_region(::CauchyPointFallback, g1::T, g2::T, h11::T, h22::T, h12::T, tr_radius::T) where T<:Real
    solve_2d_trust_region_cauchy(g1, g2, h11, h22, h12, tr_radius)
end

# Eigenvalue fallback (exact, more expensive)
function solve_2d_trust_region(::EigenvalueFallback, g1::T, g2::T, h11::T, h22::T, h12::T, tr_radius::T) where T<:Real
    solve_2d_trust_region_eigenvalue(g1, g2, h11, h22, h12, tr_radius)
end

# ============================================================================
# 2D Trust Region Solver (Cauchy point fallback - default)
# ============================================================================

function solve_2d_trust_region_cauchy(g1::T, g2::T, h11::T, h22::T, h12::T, tr_radius::T) where T<:Real
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
        
        # Check if Newton step is descent direction (model_value < 0)
        model_value = g1*s1 + g2*s2 + 0.5*(h11*s1^2 + 2*h12*s1*s2 + h22*s2^2)
        s_norm_sq = s1^2 + s2^2
        
        if model_value < 0 && s_norm_sq ≤ tr_radius^2
            # Newton step is valid descent and within TR
            return s1, s2
        elseif model_value < 0 && s_norm_sq > tr_radius^2
            # Newton step is descent but outside TR - scale to boundary
            scale = tr_radius / sqrt(s_norm_sq)
            return scale * s1, scale * s2
        end
    end
    
    # Newton step failed (not descent or H singular) - use Cauchy point
    # Cauchy point: min_{t≥0, ||ts||≤Δ} m(t*s_sd) where s_sd = -g
    # m(t*(-g)) = -t||g||^2 + 0.5*t^2*(g'*H*g)
    # Optimal t = ||g||^2 / (g'Hg) if g'Hg > 0, else t = Δ/||g||
    
    gHg = g1*g1*h11 + 2*g1*g2*h12 + g2*g2*h22
    
    if gHg > eps(T)
        # Positive curvature along gradient
        t_cauchy = g_norm_2d^2 / gHg
        t = min(t_cauchy, tr_radius / g_norm_2d)
    else
        # Non-positive curvature - go to TR boundary
        t = tr_radius / g_norm_2d
    end
    
    return -t * g1, -t * g2
end

# ============================================================================
# Eigenvalue-based 2D Solver (Exact solution via secular equation)
# ============================================================================

function solve_2d_trust_region_eigenvalue(g1::T, g2::T, h11::T, h22::T, h12::T, tr_radius::T) where T<:Real
    # Check for zero gradient
    g_norm_2d = sqrt(g1^2 + g2^2)
    if g_norm_2d < eps(T) || tr_radius == 0
        return zero(T), zero(T)
    end
    
    # Eigenvalue decomposition of 2x2 Hessian
    # For 2x2 symmetric matrix, we can compute eigenvalues analytically
    trace_H = h11 + h22
    det_H = h11 * h22 - h12^2
    discriminant = trace_H^2 - 4*det_H
    
    if discriminant < 0
        # Complex eigenvalues shouldn't happen for symmetric matrix
        # Fall back to Cauchy point
        gHg = g1*g1*h11 + 2*g1*g2*h12 + g2*g2*h22
        if gHg > eps(T)
            t_cauchy = g_norm_2d^2 / gHg
            t = min(t_cauchy, tr_radius / g_norm_2d)
        else
            t = tr_radius / g_norm_2d
        end
        return -t * g1, -t * g2
    end
    
    sqrt_disc = sqrt(max(discriminant, zero(T)))
    λ1 = (trace_H + sqrt_disc) / 2  # Larger eigenvalue
    λ2 = (trace_H - sqrt_disc) / 2  # Smaller eigenvalue
    
    # Check if positive definite
    if λ2 > 0
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
    end
    
    # Need to find boundary solution via secular equation
    # Use simple bisection or closed-form for 2D case
    # For 2D, we can solve directly using the trust region constraint
    
    # Solve ||s(λ)||² = Δ² where s(λ) solves (H + λI)s = -g
    # This is a 1D root-finding problem
    
    λ_min = -λ2  # Minimum valid λ (makes H + λI positive semidefinite)
    
    # Try a few λ values to find the boundary solution
    function secular_eq(λ::T)
        # Solve (H + λI)s = -g
        h11_λ = h11 + λ
        h22_λ = h22 + λ
        det_λ = h11_λ * h22_λ - h12^2
        
        if abs(det_λ) < eps(T)
            return Inf
        end
        
        inv_det = one(T) / det_λ
        s1 = -inv_det * (h22_λ * g1 - h12 * g2)
        s2 = -inv_det * (-h12 * g1 + h11_λ * g2)
        s_norm = sqrt(s1^2 + s2^2)
        
        return s_norm - tr_radius
    end
    
    # Binary search for λ
    λ_low = max(λ_min, zero(T))
    λ_high = λ_low + 1.0
    
    # Find upper bound
    maxiter = 50
    for _ in 1:maxiter
        if secular_eq(λ_high) < 0
            λ_high *= 2
        else
            break
        end
    end
    
    # Bisection
    for _ in 1:maxiter
        λ_mid = (λ_low + λ_high) / 2
        res = secular_eq(λ_mid)
        
        if abs(res) < 1e-10
            break
        end
        
        if res > 0
            λ_low = λ_mid
        else
            λ_high = λ_mid
        end
    end
    
    λ_sol = (λ_low + λ_high) / 2
    
    # Compute final solution
    h11_λ = h11 + λ_sol
    h22_λ = h22 + λ_sol
    det_λ = h11_λ * h22_λ - h12^2
    
    if abs(det_λ) > eps(T)
        inv_det = one(T) / det_λ
        s1 = -inv_det * (h22_λ * g1 - h12 * g2)
        s2 = -inv_det * (-h12 * g1 + h11_λ * g2)
        return s1, s2
    else
        # Fallback to Cauchy point
        gHg = g1*g1*h11 + 2*g1*g2*h12 + g2*g2*h22
        if gHg > eps(T)
            t_cauchy = g_norm_2d^2 / gHg
            t = min(t_cauchy, tr_radius / g_norm_2d)
        else
            t = tr_radius / g_norm_2d
        end
        return -t * g1, -t * g2
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