"""
    Trust-Region Optimization Main Loop

Main solve function implementing the interior-point trust-region reflective
algorithm of Coleman & Li (1994, 1996).
"""

# ============================================================================
# Parameter handling
# ============================================================================
# All objective functions must have signature f(x, p)
# Use NullParameters() if parameters are not needed

"""
    solve(prob, hessian_update, subspace, p=NullParameters(); options=RetroOptions(), maxiter=1000, verbose=false)

Solve a bound-constrained optimization problem using trust-region methods.

# Arguments
- `prob::RetroProblem`: The optimization problem to solve
- `hessian_update::AbstractHessianUpdate`: Hessian approximation strategy
  - `BFGSUpdate()`: Quasi-Newton BFGS (recommended for most problems)
  - `SR1Update()`: Symmetric Rank-1 (good for indefinite problems)
  - `ExactHessian()`: Compute exact Hessian via AD (expensive but accurate)
- `subspace::AbstractSubspace`: Trust-region subproblem solver
  - `TwoDimSubspace()`: 2D subspace method (good balance, default)
  - `CGSubspace([maxiter])`: Conjugate gradient (good for large problems)
  - `FullSpace()`: Full-dimensional solve (accurate but expensive)
- `p`: Parameters to pass to objective function (default: `NullParameters()`)
  - The objective function must have signature `f(x, p)`
  - Use `f(x, _) = ...` if parameters are not needed
  - Pass actual parameters as a named tuple, struct, or any type

# Keyword Arguments
- `maxiter::Int`: Maximum iterations (default: 1000)
- `verbose::Bool`: Print iteration info (default: false)
- `options::RetroOptions`: Algorithm parameters and tolerances

# Returns
- `RetroResult`: Contains solution, convergence info, and statistics

# Examples
```julia
# Without parameters (use _ to ignore parameter argument)
f(x, _) = sum(abs2, x)
prob = RetroProblem(f, [1.0, 2.0], AutoForwardDiff())
result = solve(prob, BFGSUpdate(), TwoDimSubspace())

# With parameters
f(x, p) = sum(abs2, x .- p.target)
prob = RetroProblem(f, [1.0, 2.0], AutoForwardDiff())
result = solve(prob, BFGSUpdate(), TwoDimSubspace(), (target=[0.0, 0.0],))

# Rosenbrock with bounds
f(x, _) = 100(x[2] - x[1]^2)^2 + (1 - x[1])^2
prob = RetroProblem(f, [-1.2, 1.0], AutoForwardDiff(); lb=[-2.0, -2.0], ub=[2.0, 2.0])
result = solve(prob, BFGSUpdate(), TwoDimSubspace(); verbose=true)
```
"""
function solve(
    prob::RetroProblem,
    hessian_update::AbstractHessianUpdate,
    subspace::AbstractSubspace,
    p = NullParameters();
    maxiter::Int = 1000,
    verbose::Bool = false,
    options::RetroOptions = RetroOptions()
)
    T = eltype(prob.x0)
    n = length(prob.x0)

    # Project initial point to feasible region
    check_and_project_bounds!(prob.x0, prob.lb, prob.ub)

    # Initialize optimizer state
    state = initialize_state(prob, prob.x0, hessian_update, options, p)

    # Identify active constraints and compute free gradient
    update_active_set!(state, options)

    # Check initial convergence
    gnorm = norm(state.gx_free, Inf)
    if verbose
        log_header()
        log_step(0, state.value, gnorm, state.tr_radius, zero(T), zero(T), true)
    end
    
    if gnorm ≤ options.gtol_a
        return RetroResult(
            state.x, state.value, copy(state.grad), 0,
            state.f_evals, state.g_evals, state.h_evals,
            true, :gtol
        )
    end

    # Main optimization loop
    old_fx = state.value
    rejected_steps = 0
    max_consecutive_rejections = 10

    for iter in 1:maxiter
        state.iter = iter

        # Solve trust-region subproblem
        solve_subproblem!(state, subspace)
        subproblem_step_norm = state.last_step_norm

        # Apply reflective bounds to step
        apply_reflective_bounds!(state, options)

        # Evaluate trial point (reusing pre-allocated buffers)
        state.x_trial .= state.x .+ state.step_reflected
        evaluate_trial_point!(state, prob, hessian_update, p)

        # Compute reduction ratio
        rho = compute_reduction_ratio(state, state.fx_trial[], state.grad_trial, subproblem_step_norm)

        # Update trust-region radius and track if it changed
        old_radius = state.tr_radius
        update_trust_region_radius!(state, rho, subproblem_step_norm, options)
        radius_updated = (state.tr_radius != old_radius)
        
        # Track radius updates for hybrid strategy (dispatched)
        track_radius_update!(hessian_update, radius_updated)

        # Decide whether to accept step
        accepted = rho > options.mu

        if accepted
            rejected_steps = 0
            
            # Update Hessian before accepting step
            update_hessian_at_trial!(state, prob, hessian_update, p)
            
            # Accept the step
            state.x .= state.x_trial
            state.value = state.fx_trial[]
            state.grad .= state.grad_trial
            state.f_evals += 1
            state.g_evals += 1

            # Update active set
            update_active_set!(state, options)

            # Compute function value change for convergence check
            fx_change = abs(state.value - old_fx)
            old_fx = state.value

            if verbose
                log_step(iter, state.value, norm(state.gx_free, Inf), state.tr_radius, 
                        norm(state.step_reflected), rho, accepted)
            end

            # Check convergence
            conv_result = check_convergence(state, iter, options, fx_change, norm(state.step_reflected))
            if conv_result !== nothing
                if verbose
                    println("Optimization terminated: $(conv_result.termination_reason)")
                end
                return conv_result
            end
        else
            # Step rejected
            rejected_steps += 1
            
            if verbose
                log_step(iter, old_fx, norm(state.gx_free, Inf), state.tr_radius,
                        norm(state.step_reflected), rho, accepted)
            end

            # Check for stagnation
            if rejected_steps >= max_consecutive_rejections
                if verbose
                    println("Optimization stagnated: $rejected_steps consecutive rejections")
                end
                return RetroResult(
                    state.x, state.value, copy(state.grad), iter,
                    state.f_evals, state.g_evals, state.h_evals,
                    false, :stagnation
                )
            end
        end

        # Check if trust region became too small
        min_tr_radius = max(options.xtol, sqrt(eps(T)) * options.initial_tr_radius)
        if state.tr_radius < min_tr_radius
            if verbose
                println("Trust region radius too small: $(state.tr_radius)")
            end
            return RetroResult(
                state.x, state.value, copy(state.grad), iter,
                state.f_evals, state.g_evals, state.h_evals,
                false, :tr_radius_too_small
            )
        end
    end

    # Maximum iterations reached
    if verbose
        println("Maximum iterations reached: $maxiter")
    end
    return RetroResult(
        state.x, state.value, copy(state.grad), maxiter,
        state.f_evals, state.g_evals, state.h_evals,
        false, :maxiter
    )
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
    track_radius_update!(hessian_update, radius_updated)

Track trust-region radius updates for hybrid strategies.
Default is no-op for non-hybrid updates.
"""
track_radius_update!(::AbstractHessianUpdate, radius_updated::Bool) = nothing

function track_radius_update!(hybrid::HybridUpdate, radius_updated::Bool)
    if radius_updated
        reset_hybrid_on_radius_update!(hybrid)
    else
        record_no_radius_update!(hybrid)
    end
end

"""
    evaluate_trial_point!(state, prob, hessian_update, p)

Evaluate function and gradient at trial point, respecting prep object type.
Writes results to state.fx_trial and state.grad_trial.
"""
function evaluate_trial_point!(state, prob, gn::GaussNewtonUpdate, p)
    # For Gauss-Newton, recompute from residuals
    # Use Constant(p) to treat p as constant during differentiation
    r = gn.resfun(state.x_trial, p)
    prep_jac = prepare_jacobian(gn.resfun, prob.adtype, state.x_trial, Constant(p))
    _, jac = value_and_jacobian(gn.resfun, prep_jac, prob.adtype, state.x_trial, Constant(p))
    
    state.fx_trial[] = 0.5 * dot(r, r)
    mul!(state.grad_trial, jac', r)
end

function evaluate_trial_point!(state, prob, hybrid::HybridUpdate, p)
    # Evaluate using the current active strategy
    strategy = current_strategy(hybrid)
    
    # Special handling: if strategy is quasi-Newton but initial was not
    if strategy isa ApproximatingHessianUpdate && !(hybrid.initial isa ApproximatingHessianUpdate)
        # Switched to quasi-Newton from exact/GN - recompute without prep
        state.fx_trial[], state.grad_trial = value_and_gradient(prob.f, prob.adtype, state.x_trial, Constant(p))
        return
    end
    
    evaluate_trial_point!(state, prob, strategy, p)
end

function evaluate_trial_point!(state, prob, ::ExactHessian, p)
    # For exact Hessian, prep is for Hessian computation - need separate gradient eval
    # Use Constant(p) to treat p as constant during differentiation
    state.fx_trial[], state.grad_trial = value_and_gradient(prob.f, prob.adtype, state.x_trial, Constant(p))
end

function evaluate_trial_point!(state, prob, ::ApproximatingHessianUpdate, p)
    # For quasi-Newton, use stored prep with Constant(params) for consistency
    # The prep was created with the same Constant(params), so this is consistent
    state.fx_trial[], state.grad_trial = value_and_gradient(prob.f, state.prep, prob.adtype, state.x_trial, Constant(state.params))
end

"""
    update_hessian_at_trial!(state, prob, hessian_update, p)

Update Hessian approximation at the accepted trial point.
"""
function update_hessian_at_trial!(state, prob, ::ExactHessian, p)
    # Compute exact Hessian at new point using stored params and prep
    # The prep was created with Constant(params), so use the same
    _, _, hess_new = value_gradient_and_hessian(prob.f, state.prep, prob.adtype, state.x_trial, Constant(state.params))
    state.hessian .= hess_new
    state.h_evals += 1
end

function update_hessian_at_trial!(state, prob, gn::GaussNewtonUpdate, p)
    # Compute Gauss-Newton Hessian at new point
    # Use Constant(p) to treat p as constant during differentiation
    _, _, hess_new = compute_gauss_newton_hessian(gn.resfun, prob.adtype, state.x_trial, Constant(p))
    state.hessian .= hess_new
    state.h_evals += 1
end

function update_hessian_at_trial!(state, prob, update::ApproximatingHessianUpdate, p)
    # Quasi-Newton update
    update_hessian_approx!(state, update)
end

function update_hessian_at_trial!(state, prob, hybrid::HybridUpdate, p)
    # Use current active strategy
    strategy = current_strategy(hybrid)
    
    if strategy isa ApproximatingHessianUpdate
        # Quasi-Newton: always update the approximation
        update_hessian_approx!(state, strategy)
    else
        # Exact or Gauss-Newton: recompute
        update_hessian_at_trial!(state, prob, strategy, p)
    end
end

"""
    initialize_state(prob, x0, hessian_update, options)

Initialize the optimizer state based on the Hessian approximation strategy.
"""
function initialize_state(prob::RetroProblem, x0, gn::GaussNewtonUpdate, options, p)
    T = eltype(x0)
    n = length(x0)
    
    # Compute initial Gauss-Newton Hessian
    # Use Constant(p) to tell AD that p should be treated as constant
    y, grad, hess = compute_gauss_newton_hessian(gn.resfun, prob.adtype, x0, Constant(p))
    
    # Note: We don't use prepare_jacobian as prep since it's only for Gauss-Newton
    # For gradient evals of trial points, we'll recompute from residuals
    prep = gn  # Store the update object itself as "prep"
    
    state = TrustRegionState(x0, prep, y, grad, hess,
                            options.initial_tr_radius, prob.lb, prob.ub, p)
    state.h_evals = 1
    
    return state
end

function initialize_state(prob::RetroProblem, x0, hybrid::HybridUpdate, options, p)
    # Initialize using the initial strategy
    # But if fallback is quasi-Newton, also prepare gradient evaluation
    state = initialize_state(prob, x0, hybrid.initial, options, p)
    
    # If the fallback uses quasi-Newton, we need a gradient prep stored somewhere
    # We'll store it in a special field or recompute when needed
    # For now, we'll handle this in evaluate_trial_point
    
    return state
end

function initialize_state(prob::RetroProblem, x0, ::ExactHessianUpdate, options, p)
    T = eltype(x0)
    n = length(x0)
    
    # Compute initial value, gradient, and Hessian
    # Use Constant(p) to tell AD that p should be treated as constant
    prep = prepare_hessian(prob.f, prob.adtype, x0, Constant(p))
    y, grad, hess = value_gradient_and_hessian(prob.f, prep, prob.adtype, x0, Constant(p))

    state = TrustRegionState(x0, prep, y, grad, hess,
                            options.initial_tr_radius, prob.lb, prob.ub, p)
    state.h_evals = 1
    
    return state
end

function initialize_state(prob::RetroProblem, x0, ::ApproximatingHessianUpdate, options, p)
    T = eltype(x0)
    n = length(x0)
    
    # Compute initial value and gradient
    # Use Constant(p) to tell AD that p should be treated as constant
    prep = prepare_gradient(prob.f, prob.adtype, x0, Constant(p))
    y, grad = value_and_gradient(prob.f, prep, prob.adtype, x0, Constant(p))
    
    # Initialize Hessian approximation as identity
    hess = Matrix{T}(I, n, n)
    
    state = TrustRegionState(x0, prep, y, grad, hess,
                            options.initial_tr_radius, prob.lb, prob.ub, p)
    
    return state
end
"""
    compute_reduction_ratio(state, fx_trial, grad_trial, step_norm)

Compute ratio of actual to predicted reduction.

Following Fides (minimize.py line 564), the actual reduction includes an augmentation term:
  aug = 0.5 * ss' * diag(dv .* |grad_trial|) * ss
  actual_reduction = f_old - f_new - aug

This augmentation accounts for the barrier function in Coleman-Li scaling.
"""
function compute_reduction_ratio(state::TrustRegionState{T}, fx_trial::T, grad_trial::AbstractVector{T}, step_norm::T) where T
    # Actual reduction
    actual_reduction = state.value - fx_trial
    
    # Use predicted reduction computed in scaled space by the subproblem solver
    if state.predicted_reduction > eps(T)
        return actual_reduction / state.predicted_reduction
    else
        return -one(T)
    end
end

"""
    update_trust_region_radius!(state, rho, step_norm, options)

Update trust region radius based on reduction ratio and step size.
"""
function update_trust_region_radius!(
    state::TrustRegionState{T},
    rho::T,
    step_norm::T,
    options
) where T
    
    interior_solution = step_norm < T(0.9) * state.tr_radius
    
    if rho >= options.eta && !interior_solution
        # Excellent agreement and step hit boundary - expand
        state.tr_radius = min(options.gamma2 * state.tr_radius, options.max_tr_radius)
    elseif rho <= options.mu
        # Poor agreement - shrink
        state.tr_radius = min(options.gamma1 * state.tr_radius, step_norm / 4)
    end
    # Otherwise keep current radius
end

"""
    update_hessian_approx!(state, update_type)

Update quasi-Newton Hessian approximation.
"""
function update_hessian_approx!(::TrustRegionState, ::ExactHessianUpdate)
    # No-op: Hessian is computed exactly at each iteration
    nothing
end

function update_hessian_approx!(state::TrustRegionState, ::BFGSUpdate)
    # Use the actual taken step (reflected), not the subproblem step
    # This is the step from old x to new x
    s = state.step_reflected
    y = state.Δg
    @. y = state.grad_trial - state.grad
    
    sy = dot(s, y)
    T = eltype(s)
    
    if sy > eps(T) * norm(s) * norm(y)
        H = state.hessian
        mul!(state.Hs, H, s)
        sHs = dot(s, state.Hs)
        
        if sHs > eps(T)
            # BFGS update: H_new = H - (Hs)(Hs)'/sHs + yy'/sy
            H .-= (state.Hs * state.Hs') ./ sHs
            H .+= (y * y') ./ sy
        end
    end
end

function update_hessian_approx!(state::TrustRegionState, ::SR1Update)
    # Use the actual taken step (reflected), not the subproblem step
    s = state.step_reflected
    y = state.Δg
    @. y = state.grad_trial - state.grad
    
    H = state.hessian
    mul!(state.Hs, H, s)
    r = y .- state.Hs
    rs = dot(r, s)
    T = eltype(s)
    
    if abs(rs) > eps(T) * norm(r) * norm(s)
        # SR1 update: H_new = H + rr'/rs
        H .+= (r * r') ./ rs
    end
end

"""
    check_convergence(state, iter, options, fx_change, step_norm)

Check convergence criteria.
"""
function check_convergence(
    state::TrustRegionState{T},
    iter::Int,
    options,
    fx_change::T,
    step_norm::T
) where T
    
    gnorm = norm(state.gx_free, Inf)
    current_fx = state.value
    
    # Gradient convergence (primary criterion)
    if gnorm ≤ options.gtol_a
        return RetroResult(
            state.x, current_fx, copy(state.grad), iter,
            state.f_evals, state.g_evals, state.h_evals,
            true, :gtol
        )
    end
    
    # Relative gradient convergence
    if options.gtol_r > 0 && gnorm ≤ options.gtol_r * abs(current_fx)
        return RetroResult(
            state.x, current_fx, copy(state.grad), iter,
            state.f_evals, state.g_evals, state.h_evals,
            true, :gtol
        )
    end
    
    # Function value convergence
    if options.ftol_a > 0 || options.ftol_r > 0
        ftol_abs = fx_change
        ftol_rel = fx_change / (abs(current_fx) + eps(T))
        
        if (options.ftol_a > 0 && ftol_abs ≤ options.ftol_a) || 
           (options.ftol_r > 0 && ftol_rel ≤ options.ftol_r)
            # Only declare convergence if gradient is also reasonably small
            if gnorm ≤ T(100) * options.gtol_a
                return RetroResult(
                    state.x, current_fx, copy(state.grad), iter,
                    state.f_evals, state.g_evals, state.h_evals,
                    true, :ftol
                )
            end
        end
    end
    
    # Step size convergence
    if options.xtol > 0 && step_norm ≤ options.xtol
        if gnorm ≤ T(100) * options.gtol_a
            return RetroResult(
                state.x, current_fx, copy(state.grad), iter,
                state.f_evals, state.g_evals, state.h_evals,
                true, :xtol
            )
        end
    end
    
    return nothing
end

# ============================================================================
# Logging Functions
# ============================================================================

function log_header()
    println("iter |    fval    |  ||g||∞  | tr_radius | ||step|| |   ρ   | acc")
    println("-----|------------|----------|-----------|----------|-------|----")
end

function log_step(iter, fval, gnorm, tr_radius, step_norm, rho, accepted)
    iter_str = lpad(iter, 4)
    fval_str = @sprintf("%.2E", fval)
    gnorm_str = @sprintf("%.2E", gnorm)
    tr_str = @sprintf("%.2E", tr_radius)
    step_str = iter > 0 ? @sprintf("%.2E", step_norm) : "  ---   "
    rho_str = iter > 0 ? @sprintf("%+.2f", rho) : " ---  "
    acc_str = accepted ? " ✓ " : " ✗ "
    
    println("$iter_str | $fval_str | $gnorm_str | $tr_str | $step_str | $rho_str | $acc_str")
end
