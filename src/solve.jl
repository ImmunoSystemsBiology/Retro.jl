function solve(
    prob::FidesProblem, 
    hessian_update::AbstractHessianUpdate,
    subproblem_solver::AbstractSubproblemSolver;
    options::TrustRegionOptions = TrustRegionOptions()
)
    T = eltype(prob.x0)
    n = length(prob.x0)
    
    # Project initial point
    x0 = project_bounds(prob.x0, prob.lb, prob.ub)
    
    # Initialize based on Hessian type
    state = initialize_state(prob, x0, hessian_update, options)
    
    # Update active set
    update_active_set!(state, options)
    
    # Check initial convergence
    gnorm = norm(state.gx_free, Inf)
    if gnorm ≤ options.gtol
        return TrustRegionResult(
            state.x, fx(state), copy(gx(state)), 0, 
            state.f_evals, state.g_evals, state.h_evals, 
            true, :gtol
        )
    end
    
    if options.verbose
        println("Initial: f = $(fx(state)), ||g_free||∞ = $gnorm, Δ = $(state.tr_radius)")
    end
    
    # Main loop
    for iter in 1:options.maxiter
        state.iter = iter
        
        # Solve subproblem and get step
        solve_subproblem!(state, subproblem_solver)
        subproblem_step_norm = state.last_step_norm
        
        # Apply reflective bounds
        apply_reflective_bounds!(state, options)
        
        # Evaluate trial point
        evaluate_trial_point!(state, prob.f, prob.adtype)
        
        # Compute reduction ratio
        rho = compute_reduction_ratio(state)
        
        # Update trust region
        update_trust_region_radius!(state, rho, subproblem_step_norm, options)
        
        # Accept or reject step
        if rho > options.eta1
            accept_step!(state, hessian_update, prob.f, prob.adtype, options, rho)
            
            # Check convergence
            conv_result = check_convergence(state, iter, options)
            if conv_result !== nothing
                return conv_result
            end
        else
            if options.verbose
                println("Iter $iter: step rejected, ρ = $rho, Δ = $(state.tr_radius)")
            end
        end
        
        if state.tr_radius < options.xtol
            return TrustRegionResult(
                state.x, fx(state), copy(gx(state)), iter,
                state.f_evals, state.g_evals, state.h_evals, 
                false, :tr_radius_too_small
            )
        end
    end
    
    return TrustRegionResult(
        state.x, fx(state), copy(gx(state)), options.maxiter,
        state.f_evals, state.g_evals, state.h_evals, 
        false, :maxiter
    )
end

# ============================================================================
# Initialization
# ============================================================================

function initialize_state(prob::FidesProblem, x0, hessian_update::ExactHessian, options)
    T = eltype(x0)
    n = length(x0)
    
    # Create HessianResult and compute initial values
    # DifferentiationInterface signature: hessian!(f, result, backend, x)
    diff_result = DiffResults.HessianResult(x0)
    hessian!(prob.f, diff_result, prob.adtype, x0)
    
    # No separate Hessian approximation needed
    Hx_approx = nothing
    
    state = TrustRegionState(x0, diff_result, Hx_approx, 
                            options.initial_tr_radius, prob.lb, prob.ub)
    state.h_evals = 1
    
    return state
end

function initialize_state(prob::FidesProblem, x0, 
                         hessian_update::Union{BFGSUpdate, SR1Update}, options)
    T = eltype(x0)
    n = length(x0)
    
    # Create GradientResult and compute initial values
    # DifferentiationInterface signature: gradient!(f, result, backend, x)
    diff_result = DiffResults.GradientResult(x0)
    gradient!(prob.f, diff_result, prob.adtype, x0)
    
    # Initialize Hessian approximation as identity
    Hx_approx = Matrix{T}(I, n, n)
    
    state = TrustRegionState(x0, diff_result, Hx_approx,
                            options.initial_tr_radius, prob.lb, prob.ub)
    
    return state
end

# ============================================================================
# Utilities
# ============================================================================

function compute_predicted_reduction(state::TrustRegionState)
    s = state.step_reflected
    g = gx(state)  # Use accessor
    H = Hx(state)  # Use accessor
    
    mul!(state.Hs, H, s)
    return -(dot(g, s) + 0.5 * dot(s, state.Hs))
end

function update_trust_region_radius!(state::TrustRegionState{T}, rho::T, 
                                    subproblem_step_norm::T, options) where T
    if rho < options.eta1
        # Poor agreement - shrink trust region
        state.tr_radius *= options.gamma1
    elseif rho > options.eta2 && subproblem_step_norm ≥ T(0.95) * state.tr_radius
        # Excellent agreement AND step hit boundary - expand trust region
        state.tr_radius = min(options.gamma2 * state.tr_radius, options.max_tr_radius)
    elseif rho > T(0.5) && subproblem_step_norm ≥ T(0.95) * state.tr_radius
        # Good agreement and at boundary - moderately expand
        state.tr_radius = min(T(1.5) * state.tr_radius, options.max_tr_radius)
    end
    # Otherwise: acceptable step but not at boundary - keep current radius
end

# ============================================================================
# Helper Functions for Main Loop
# ============================================================================

"""Evaluate trial point and compute gradient"""
function evaluate_trial_point!(state::TrustRegionState, f, adtype)
    state.x_trial .= state.x .+ state.step_reflected
    gradient!(f, state.diff_result_trial, adtype, state.x_trial)
    state.f_evals += 1
    state.g_evals += 1
end

"""Compute reduction ratio"""
function compute_reduction_ratio(state::TrustRegionState{T}) where T
    pred_reduction = compute_predicted_reduction(state)
    actual_reduction = fx(state) - fx_trial(state)
    return pred_reduction > eps(T) ? actual_reduction / pred_reduction : -one(T)
end

"""Accept step and update state"""
function accept_step!(state::TrustRegionState, hessian_update, f, adtype, options, rho)
    old_fx = fx(state)
    state.x .= state.x_trial
    
    if hessian_update isa ExactHessian
        # For exact Hessian, compute at new point (overwrites diff_result completely)
        # Then copy gradient from diff_result to have consistent state
        update_hessian!(state, hessian_update, f, adtype)
        # No swap - diff_result now has everything at new point
    else
        # For quasi-Newton: update BEFORE swap (needs old and new gradients)
        update_hessian!(state, hessian_update, f, adtype)
        # Swap DiffResults to avoid copying
        state.diff_result, state.diff_result_trial = 
            state.diff_result_trial, state.diff_result
    end
    
    # Update active constraints
    update_active_set!(state, options)
    
    # Verbose output
    if options.verbose
        gnorm = norm(state.gx_free, Inf)
        step_norm = state.last_step_norm
        active_count = count(state.active_set)
        println("Iter $(state.iter): f = $(fx(state)), ||g_free||∞ = $gnorm, ||s|| = $step_norm, Δ = $(state.tr_radius), ρ = $rho, active = $active_count")
    end
end

"""Check convergence criteria"""
function check_convergence(state::TrustRegionState{T}, iter::Int, options) where T
    gnorm = norm(state.gx_free, Inf)
    step_norm = state.last_step_norm
    
    # Gradient convergence
    if gnorm ≤ options.gtol
        return TrustRegionResult(
            state.x, fx(state), copy(gx(state)), iter, 
            state.f_evals, state.g_evals, state.h_evals, 
            true, :gtol
        )
    end
    
    # Step size convergence
    if step_norm ≤ options.xtol && options.xtol > 0
        return TrustRegionResult(
            state.x, fx(state), copy(gx(state)), iter,
            state.f_evals, state.g_evals, state.h_evals, 
            true, :xtol
        )
    end
    
    return nothing  # No convergence yet
end

# ============================================================================
# Additional Utility Functions
# ============================================================================

"""
    is_feasible(x, lb, ub)

Check if point x satisfies bound constraints.
"""
function is_feasible(x::AbstractVector, lb, ub)
    if lb !== nothing && any(x .< lb .- 1e-12)
        return false
    end
    if ub !== nothing && any(x .> ub .+ 1e-12)  
        return false
    end
    return true
end

"""
    distance_to_bounds(x, lb, ub)

Compute minimum distance from x to bound constraints.
"""
function distance_to_bounds(x::AbstractVector{T}, lb, ub) where T<:Real
    min_dist = T(Inf)
    
    if lb !== nothing
        for i in eachindex(x)
            min_dist = min(min_dist, x[i] - lb[i])
        end
    end
    
    if ub !== nothing
        for i in eachindex(x)
            min_dist = min(min_dist, ub[i] - x[i])
        end
    end
    
    return max(min_dist, zero(T))
end

"""
    compute_cauchy_point(state, tr_radius)

Compute Cauchy point for trust region subproblem with bounds.
"""
function compute_cauchy_point(state::TrustRegionState{T}, tr_radius::T) where T<:Real
    g_free = state.gx_free
    H = state.Hx
    
    gnorm = norm(g_free)
    if gnorm == 0
        return zeros(T, length(g_free))
    end
    
    # Compute Hg for free variables only
    Hg_free = H * g_free
    gHg = dot(g_free, Hg_free)
    
    if gHg > eps(T)
        alpha = gnorm^2 / gHg
    else
        alpha = tr_radius / gnorm
    end
    
    # Project onto trust region
    alpha = min(alpha, tr_radius / gnorm)
    
    cauchy_step = -alpha .* g_free
    
    # Handle bound constraints - find intersection with bounds
    if state.lb !== nothing || state.ub !== nothing
        alpha_bound = one(T)
        
        for i in eachindex(cauchy_step)
            if state.active_set[i]
                continue  # Skip active variables
            end
            
            if state.lb !== nothing && cauchy_step[i] < 0
                alpha_i = (state.lb[i] - state.x[i]) / cauchy_step[i]
                alpha_bound = min(alpha_bound, alpha_i)
            end
            
            if state.ub !== nothing && cauchy_step[i] > 0
                alpha_i = (state.ub[i] - state.x[i]) / cauchy_step[i]
                alpha_bound = min(alpha_bound, alpha_i)
            end
        end
        
        cauchy_step .*= max(alpha_bound, zero(T))
    end
    
    return cauchy_step
end

"""
    compute_gradient_norm_inf(state)

Compute infinity norm of projected gradient.
"""
function compute_gradient_norm_inf(state::TrustRegionState)
    return norm(state.gx_free, Inf)
end

"""
    check_optimality_conditions(state, options)

Check first-order optimality conditions for bound-constrained problem.
"""
function check_optimality_conditions(state::TrustRegionState{T}, options) where T<:Real
    x, g = state.x, state.gx
    lb, ub = state.lb, state.ub
    
    # Check KKT conditions
    for i in eachindex(x)
        if lb !== nothing && x[i] ≈ lb[i]
            # At lower bound: gradient should be non-negative
            if g[i] < -options.gtol
                return false
            end
        elseif ub !== nothing && x[i] ≈ ub[i]
            # At upper bound: gradient should be non-positive
            if g[i] > options.gtol
                return false
            end
        else
            # Interior point: gradient should be near zero
            if abs(g[i]) > options.gtol
                return false
            end
        end
    end
    
    return true
end

"""
    regularize_hessian(H, min_eigenvalue)

Regularize Hessian matrix to ensure positive definiteness.
"""
function regularize_hessian(H::AbstractMatrix{T}, min_eigenvalue::T = T(1e-8)) where T<:Real
    try
        # Check if already positive definite
        cholesky(H)
        return H, zero(T)
    catch
        # Need regularization
        E = eigen(H)
        lambda_min = minimum(E.values)
        
        if lambda_min < min_eigenvalue
            reg_param = min_eigenvalue - lambda_min
            return H + reg_param * I, reg_param
        else
            return H, zero(T)
        end
    end
end

"""
    compute_model_reduction(step, g, H)

Compute predicted reduction in quadratic model.
"""
function compute_model_reduction(step::AbstractVector{T}, g::AbstractVector{T}, H::AbstractMatrix{T}) where T<:Real
    return -(dot(g, step) + T(0.5) * dot(step, H * step))
end

"""
    line_search_backtrack(f, x, step, fx, gx; alpha0=1.0, rho=0.5, c1=1e-4)

Simple backtracking line search for fallback.
"""
function line_search_backtrack(f, x::AbstractVector{T}, step::AbstractVector{T}, 
                              fx::T, gx::AbstractVector{T};
                              alpha0::T = T(1.0), rho::T = T(0.5), c1::T = T(1e-4)) where T<:Real
    alpha = alpha0
    directional_derivative = dot(gx, step)
    
    for _ in 1:20  # Max backtracking steps
        x_new = x + alpha * step
        fx_new = f(x_new)
        
        # Armijo condition
        if fx_new ≤ fx + c1 * alpha * directional_derivative
            return alpha, fx_new
        end
        
        alpha *= rho
    end
    
    return alpha, f(x + alpha * step)
end