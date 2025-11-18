# ============================================================================
# Subproblem Solvers
# ============================================================================

function solve_subproblem!(
    state::TrustRegionState{T, VT, MT},
    solver::AbstractSubproblemSolver,
) where {T, VT, MT}
    solver(state)
end

function (::TwoDimSubspace)(state::TrustRegionState{T}) where {T<:Real}
    d1, d2, g1, g2, h11, h22, h12, tr_radius = init_2d_trust_region!(state)

    # Solve exactly using eigenvalue method
    alpha, beta = solve_2d_trust_region(g1, g2, h11, h22, h12, tr_radius)
    
    state.step .= alpha .* d1 .+ beta .* d2
    
    # Zero out steps for active variables
    state.step[state.active_set] .= zero(T)
end

function (solver::CGSubspace)(state::TrustRegionState)
    # Conjugate Gradient using Steihaug's method
    steihaug_cg!(state, solver.maxiter)
end

function (::FullSpace)(state::TrustRegionState)
    # Full space solver using exact eigenvalue method
    full_space_solver!(state)
end

function init_2d_trust_region!(state::TrustRegionState{T}) where T<:Real
    g_free = state.gx_free
    H = state.Hx
    tr_radius = state.tr_radius
    
    gnorm = norm(g_free)
    if gnorm == 0
        fill!(state.step, zero(T))
        return
    end
    
    # First direction: steepest descent (free gradient)
    d1 = -g_free ./ gnorm
    
    # Second direction: Newton direction (if possible)
    state.Hg .= H * g_free
    Hg_norm = norm(state.Hg)
    
    if Hg_norm > eps(T)
        d2 = -state.Hg ./ Hg_norm
    else
        # Use random direction orthogonal to d1
        d2 = randn(T, length(g_free))
        d2 .-= dot(d2, d1) .* d1
        d2 ./= norm(d2)
    end
    
    # Solve 2D subproblem in span(d1, d2)
    # min g^T(α*d1 + β*d2) + 0.5*(α*d1 + β*d2)^T*H*(α*d1 + β*d2)
    # subject to α² + β² ≤ Δ²
    
    g1 = dot(g_free, d1)
    g2 = dot(g_free, d2)
    h11 = dot(d1, H * d1)
    h22 = dot(d2, H * d2) 
    h12 = dot(d1, H * d2)

    return d1, d2, g1, g2, h11, h22, h12, tr_radius
end

function solve_2d_trust_region(g1, g2, h11, h22, h12, tr_radius)
    # Solve 2D trust region subproblem exactly
    # Form the 2x2 Hessian
    H2d = [h11 h12; h12 h22]
    g2d = [g1; g2]
    
    try
        # Try Newton step first
        step_newton = -H2d \ g2d
        if norm(step_newton) ≤ tr_radius
            return step_newton[1], step_newton[2]
        end
    catch
        # Hessian might be singular
    end
    
    # Solve constrained problem using Lagrange multipliers
    # (H + λI)s = -g, ||s|| = Δ
    
    function secular(lambda)
        try
            H_reg = H2d + lambda * I
            s = -H_reg \ g2d
            return norm(s)^2 - tr_radius^2
        catch
            return Inf
        end
    end
    
    # Find lambda using bisection
    lambda_low = 0.0
    lambda_high = 100.0
    
    while secular(lambda_high) > 0
        lambda_high *= 2
    end
    
    for _ in 1:50  # Bisection iterations
        lambda_mid = (lambda_low + lambda_high) / 2
        if abs(secular(lambda_mid)) < 1e-12
            break
        end
        if secular(lambda_mid) > 0
            lambda_low = lambda_mid
        else
            lambda_high = lambda_mid
        end
    end
    
    lambda = (lambda_low + lambda_high) / 2
    s = -(H2d + lambda * I) \ g2d
    return s[1], s[2]
end

function steihaug_cg!(state::TrustRegionState{T}, maxiter::Int) where T
    g_free = state.gx_free
    H = state.Hx
    tr_radius = state.tr_radius
    
    fill!(state.step, zero(T))
    
    if norm(g_free) == 0
        return
    end
    
    # Initialize CG
    r = copy(g_free)  # residual
    p = -r            # search direction
    rsold = dot(r, r)
    
    for i in 1:maxiter
        Hp = H * p
        pHp = dot(p, Hp)
        
        if pHp ≤ 0
            # Negative curvature detected
            # Find intersection with trust region boundary
            step_norm = norm(state.step)
            alpha = solve_quadratic_tr(state.step, p, tr_radius)
            state.step .+= alpha .* p
            break
        end
        
        alpha = rsold / pHp
        state.step .+= alpha .* p
        
        # Check trust region constraint
        if norm(state.step) ≥ tr_radius
            # Backtrack to trust region boundary
            state.step .-= alpha .* p
            alpha = solve_quadratic_tr(state.step, p, tr_radius)
            state.step .+= alpha .* p
            break
        end
        
        r .+= alpha .* Hp
        rsnew = dot(r, r)
        
        if sqrt(rsnew) < 1e-10 * norm(g_free)
            break
        end
        
        beta = rsnew / rsold
        p .= -r .+ beta .* p
        rsold = rsnew
    end
    
    # Zero out steps for active variables
    state.step[state.active_set] .= zero(T)
end

function solve_quadratic_tr(s, p, tr_radius)
    # Solve ||s + α*p||² = tr_radius² for α ≥ 0
    sp = dot(s, p)
    pp = dot(p, p)
    ss = dot(s, s)
    
    discriminant = sp^2 - pp * (ss - tr_radius^2)
    if discriminant < 0
        return 0.0
    end
    
    alpha1 = (-sp + sqrt(discriminant)) / pp
    alpha2 = (-sp - sqrt(discriminant)) / pp
    
    return max(alpha1, alpha2)
end

function full_space_solver!(state::TrustRegionState{T}) where T<:Real
    g_free = state.gx_free
    H = state.Hx
    tr_radius = state.tr_radius
    
    if norm(g_free) == 0
        fill!(state.step, zero(T))
        return
    end
    
    try
        # Try Cholesky factorization
        F = cholesky(H)
        step_newton = -(F \ g_free)
        
        if norm(step_newton) ≤ tr_radius
            state.step .= step_newton
            state.step[state.active_set] .= zero(T)
            return
        end
    catch
        # Hessian is not positive definite
    end
    
    # Use eigenvalue method for indefinite case
    try
        E = eigen(H)
        lambda_min = minimum(E.values)
        
        # Regularization parameter
        lambda = max(0.0, -lambda_min + 1e-8)
        
        # Solve regularized system iteratively
        for _ in 1:20
            H_reg = H + lambda * I
            try
                step = -H_reg \ g_free
                step_norm = norm(step)
                
                if abs(step_norm - tr_radius) / tr_radius < 1e-6
                    state.step .= step
                    break
                elseif step_norm > tr_radius
                    lambda *= 2
                else
                    lambda /= 2
                end
            catch
                lambda *= 2
            end
        end
    catch
        # Fallback to Cauchy step
        Hg = H * g_free
        gHg = dot(g_free, Hg)
        gnorm = norm(g_free)
        
        if gHg > 0
            alpha = gnorm^2 / gHg
        else
            alpha = tr_radius / gnorm
        end
        
        alpha = min(alpha, tr_radius / gnorm)
        state.step .= -alpha .* g_free
    end
    
    # Zero out steps for active variables
    state.step[state.active_set] .= zero(T)
end
