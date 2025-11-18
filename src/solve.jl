# ============================================================================
# Main Solve Function (SciML-style interface)
# ============================================================================

"""
    solve(prob::FidesProblem, hessian_update::AbstractHessianUpdate, 
          subproblem_solver::AbstractSubproblemSolver; options=TrustRegionOptions())

Solve optimization problem using trust region method with reflective bounds.

# Arguments  
- `prob`: FidesProblem containing objective, initial point, AD type, and bounds
- `hessian_update`: Hessian update strategy (BFGSUpdate(), SR1Update(), ExactHessian()) 
- `subproblem_solver`: Subproblem solver (TwoDimSubspace(), CGSubspace(), FullSpace())
- `options`: Algorithm options
"""
function solve(
    prob::FidesProblem, 
    hessian_update::AbstractHessianUpdate,
    subproblem_solver::AbstractSubproblemSolver;
    options::TrustRegionOptions = TrustRegionOptions()
)
    T = eltype(prob.x0)
    n = length(prob.x0)
    
    # Project initial point onto feasible region
    x0 = project_bounds(prob.x0, prob.lb, prob.ub)
    
    # Initialize function evaluation
    fx0 = prob.f(x0)
    gx0 = similar(x0)
    gradient!(prob.f, gx0, prob.adtype, x0)
    
    # Initialize Hessian
    if hessian_update isa ExactHessian
        Hx0 = zeros(T, n, n)
        hessian!(prob.f, Hx0, prob.adtype, x0)
        h_evals_init = 1
    else
        Hx0 = Matrix{T}(I, n, n)
        h_evals_init = 0
    end
    
    # Initialize state
    state = TrustRegionState(x0, fx0, gx0, Hx0, options.initial_tr_radius,
                            prob.lb, prob.ub)
    state.h_evals = h_evals_init
    
    # Update active set and free gradient
    update_active_set!(state, options)
    
    # Check initial convergence
    gnorm = norm(state.gx_free, Inf)
    if gnorm ≤ options.gtol
        return TrustRegionResult(
            state.x, state.fx, state.gx, 0, state.f_evals, state.g_evals, 
            state.h_evals, true, :gtol
        )
    end
    
    if options.verbose
        println("Initial: f = $(state.fx), ||g_free||∞ = $gnorm, Δ = $(state.tr_radius)")
    end
    
    # Main optimization loop
    for iter in 1:options.maxiter
        state.iter = iter
        
        # Solve trust region subproblem with bounds
        solve_subproblem!(state, subproblem_solver)
        
        # Apply reflective bounds
        apply_reflective_bounds!(state, options)
        
        # Evaluate trial point
        state.x_trial .= state.x .+ state.step_reflected
        fx_trial = prob.f(state.x_trial)
        state.f_evals += 1
        
        # Compute reduction ratio
        pred_reduction = compute_predicted_reduction(state)
        actual_reduction = state.fx - fx_trial
        rho = pred_reduction > eps(T) ? actual_reduction / pred_reduction : -one(T)
        
        # Update trust region radius
        update_trust_region_radius!(state, rho, options)
        
        # Accept or reject step
        if rho > options.eta1
            # Accept step
            state.x .= state.x_trial
            old_fx = state.fx
            state.fx = fx_trial
            
            # Compute new gradient
            gradient!(prob.f, state.g_trial, prob.adtype, state.x)
            state.g_evals += 1
            
            # Update Hessian approximation
            update_hessian!(state, hessian_update, prob.f, prob.adtype)
            
            state.gx .= state.g_trial
            update_active_set!(state, options)
            
            # Check convergence
            gnorm = norm(state.gx_free, Inf)
            step_norm = norm(state.step_reflected)
            fx_change = abs(state.fx - old_fx)
            
            if options.verbose
                active_count = count(state.active_set)
                println("Iter $iter: f = $(state.fx), ||g_free||∞ = $gnorm, ||s|| = $step_norm, Δ = $(state.tr_radius), ρ = $rho, active = $active_count")
            end
            
            # Convergence tests
            if gnorm ≤ options.gtol
                return TrustRegionResult(
                    state.x, state.fx, state.gx, iter, state.f_evals, 
                    state.g_evals, state.h_evals, true, :gtol
                )
            end
            
            if step_norm ≤ options.xtol
                return TrustRegionResult(
                    state.x, state.fx, state.gx, iter, state.f_evals,
                    state.g_evals, state.h_evals, true, :xtol  
                )
            end
            
            if fx_change ≤ options.ftol
                return TrustRegionResult(
                    state.x, state.fx, state.gx, iter, state.f_evals,
                    state.g_evals, state.h_evals, true, :ftol
                )
            end
        else
            if options.verbose
                println("Iter $iter: step rejected, ρ = $rho, Δ = $(state.tr_radius)")
            end
        end
        
        # Check if trust region became too small
        if state.tr_radius < options.xtol
            return TrustRegionResult(
                state.x, state.fx, state.gx, iter, state.f_evals,
                state.g_evals, state.h_evals, false, :tr_radius_too_small
            )
        end
    end
    
    # Maximum iterations reached
    return TrustRegionResult(
        state.x, state.fx, state.gx, options.maxiter, state.f_evals,
        state.g_evals, state.h_evals, false, :maxiter
    )
end

# ============================================================================
# Utilities
# ============================================================================

function update_trust_region_radius!(state::TrustRegionState{T}, rho::T, options) where T
    if rho < options.eta1
        state.tr_radius *= options.gamma1
    elseif rho > options.eta2 && norm(state.step_reflected) ≈ state.tr_radius
        state.tr_radius = min(options.gamma2 * state.tr_radius, options.max_tr_radius)
    end
end

function compute_predicted_reduction(state::TrustRegionState)
    s = state.step_reflected
    g = state.gx
    H = state.Hx
    
    return -(dot(g, s) + 0.5 * dot(s, H * s))
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