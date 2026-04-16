using LinearAlgebra
using Printf

"""
    RetroOptions{T<:Real}

Algorithm parameters for trust-region optimization.

# Convergence Criteria
- `xtol::T`: Step tolerance (default: 0.0, disabled)
- `ftol_a::T`: Absolute function tolerance (default: 1e-8)
- `ftol_r::T`: Relative function tolerance (default: 1e-8)
- `gtol_a::T`: Absolute gradient tolerance (default: 1e-6)
- `gtol_r::T`: Relative gradient tolerance (default: 0.0, disabled)

# Trust Region Parameters  
- `initial_tr_radius::T`: Initial trust region radius (default: 1.0)
- `max_tr_radius::T`: Maximum allowed radius (default: 1000.0)
- `mu::T`: Shrink threshold - shrink if ρ < mu (default: 0.25)
- `eta::T`: Expand threshold - expand if ρ > eta (default: 0.75)
- `gamma1::T`: Shrink factor (default: 0.25)
- `gamma2::T`: Expand factor (default: 2.0)

# Bound Constraint Parameters
- `theta1::T`: Reflection threshold for bounds (default: 0.1)
- `theta2::T`: Secondary reflection threshold (default: 0.2)

# Example
```julia
opts = RetroOptions(gtol_a=1e-6, maxiter=100)
```
"""
struct RetroOptions{T<:Real}
    # Convergence tolerances
    xtol::T
    ftol_a::T
    ftol_r::T
    gtol_a::T
    gtol_r::T
    
    # Trust region parameters
    initial_tr_radius::T
    max_tr_radius::T
    mu::T
    eta::T
    gamma1::T
    gamma2::T
    
    # Reflective bounds parameters
    theta1::T
    theta2::T
    
    function RetroOptions{T}(;
        xtol::T = zero(T), 
        gtol_a::T = T(1e-6),
        gtol_r::T = zero(T),
        ftol_a::T = T(1e-8),
        ftol_r::T = T(1e-8),
        initial_tr_radius::T = one(T),
        max_tr_radius::T = T(1000),
        mu::T = T(0.25),
        eta::T = T(0.75),
        gamma1::T = T(0.25),
        gamma2::T = T(2.0),
        theta1::T = T(0.1),
        theta2::T = T(0.2)
    ) where {T<:Real}
        new{T}(xtol, ftol_a, ftol_r, gtol_a, gtol_r, initial_tr_radius, max_tr_radius,
               mu, eta, gamma1, gamma2, theta1, theta2)
    end
end

RetroOptions(; kwargs...) = RetroOptions{Float64}(; kwargs...)

"""
    optimize(prob::RetroProblem; kwargs...)

Solve the trust-region optimization problem.

# Arguments
- `prob::RetroProblem`: The optimization problem

# Keyword Arguments
- `x0::Vector`: Initial guess (default: prob.x0)
- `maxiter::Int`: Maximum iterations (default: 1000)
- `display::AbstractDisplayMode`: Display mode (default: Silent())
- `options::RetroOptions`: Algorithm options (default: RetroOptions())
- `subspace::AbstractSubspace`: Subspace method (default: TwoDimSubspace())
- `tr_solver::AbstractTRSolver`: Trust-region solver (default: EigenTRSolver())
- `hessian_approximation::AbstractHessianApproximation`: Hessian method (default: BFGS())

# Returns
- `RetroResult`: Optimization results
"""
function optimize(
    prob::RetroProblem{OBJ,T};
    x0::T = copy(prob.x0),
    maxiter::Int = 1000,
    display::AbstractDisplayMode = Silent(),
    options::RetroOptions = RetroOptions{eltype(T)}(),
    subspace::AbstractSubspace = TwoDimSubspace(),
    tr_solver::AbstractTRSolver = EigenTRSolver{eltype(T)}(),
    hessian_approximation::AbstractHessianApproximation = BFGS{eltype(T)}()
) where {OBJ<:AbstractObjectiveFunction, T<:AbstractVector}

    ET = eltype(x0)

    cache = RetroCache{ET}(length(x0))

    hessian_state = init_hessian!(hessian_approximation, cache)
    subspace_state = init_subspace!(subspace, cache)

    x = copy(x0)
    
    if any(isfinite, prob.lb) || any(isfinite, prob.ub)
        initialize_away_from_bounds!(x, prob.lb, prob.ub)
    end
    
    Delta = options.initial_tr_radius
    
    x_norm = norm(x)
    if x_norm > eps(ET)
        Delta = min(Delta, x_norm)
    end
    
    display_header(display)
    progress = RetroProgress(maxiter, display)
    
    f_current = value_and_gradient!(cache.g, cache, prob.objective, x)
    g_norm = norm(cache.g)
    
    converged = false
    termination_reason = :maxiter
    consecutive_rejections = 0
    f_change = zero(ET) 
    
    display_iteration(display, 0, f_current, g_norm, Delta, 0.0, "Initial")
    display_debug_initial(display, x, f_current, cache.g, g_norm, Delta)
    update_progress!(progress, 0, f_current, g_norm, "Starting")
    
    has_bounds = any(isfinite, prob.lb) || any(isfinite, prob.ub)
    
    for k in 1:maxiter

        converged, termination_reason = check_convergence(cache.g, cache.p, f_change, options)
        
        if !converged && has_bounds
            pg_norm = projected_gradient_norm(cache.g, x, prob.lb, prob.ub)
            if pg_norm < options.gtol_a
                converged = true
                termination_reason = :gtol
            end
        end
        
        if converged
            return imdone(cache, x, progress, display, k, f_current, termination_reason)
        end

        try
            update_hessian!(hessian_approximation, hessian_state, cache, prob.objective, x)
        catch e
            @warn "Hessian update failed at iteration $k: $e"
        end
        
        try
            step_norm = compute_trust_region_step!(
                cache, prob, subspace, subspace_state, 
                hessian_approximation, hessian_state, 
                tr_solver, x, Delta, options
            )
            
            compute_hv_product!(cache.tmp, hessian_approximation, hessian_state, cache, cache.p)
            pred_red = predicted_reduction(cache.g, cache.p, cache.tmp)
            
            f_trial = value_and_gradient!(cache.r, cache, prob.objective, cache.x_trial)
            actual_red = actual_reduction(f_current, f_trial)
            
            # Trust-region ratio. If model predicts no decrease, force rejection.
            rho = if pred_red > eps(ET)
                actual_red / pred_red
            else
                -ET(Inf)
            end
            
            g_dot_p  = dot(cache.g, cache.p)
            p_dot_Hp = dot(cache.p, cache.tmp)
            f_before = f_current
            g_norm_before = g_norm
            
            # Accept only when both model and objective improve.
            if pred_red > eps(ET) && actual_red > zero(ET) && accept_step(rho, options.mu)
                f_prev = f_current
                copy!(x, cache.x_trial)
                f_current = f_trial
                copy!(cache.g, cache.r)
                g_norm = norm(cache.g)
                f_change = abs(f_prev - f_current)
                
                consecutive_rejections = 0
                status = "Accepted"
                
            else
                consecutive_rejections += 1
                status = "Rejected"
                
                if consecutive_rejections > 10
                    termination_reason = :stagnation
                    return imdone(cache, x, progress, display, k, f_current, termination_reason)
                end
            end

            Delta_old = Delta
            Delta = update_trust_region_radius(
                Delta, rho, step_norm, options.mu, options.eta, 
                options.gamma1, options.gamma2, options.max_tr_radius
            )
            
            x_scale = max(norm(x), one(ET))
            if Delta < eps(ET) * x_scale
                termination_reason = :tr_radius_too_small
                return imdone(cache, x, progress, display, k, f_current, termination_reason)
            end
            
            display_iteration(display, k, f_current, g_norm, Delta, rho, status)
            display_debug_info(
                display, k, x, cache.g, g_norm_before, cache.p, cache.x_trial, cache.tmp,
                step_norm, f_before, f_trial,
                pred_red, actual_red, rho,
                Delta_old, Delta, status,
                consecutive_rejections, options.mu,
                g_dot_p, p_dot_Hp
            )
            update_progress!(progress, k, f_current, g_norm, status)
            
        catch e
            @warn "Trust-region step failed at iteration $k: $e"
            termination_reason = :numerical_error
            return imdone(cache, x, progress, display, k, f_current, termination_reason)
        end
    end
    
    return imdone(cache, x, progress, display, maxiter, f_current, :maxiter)
end

function imdone(cache, x, progress, display, iter, f_current, termination_reason)
    finish_progress!(progress)
    
    result = RetroResult(
        copy(x), f_current, copy(cache.g),
        iter,
        cache.f_calls, cache.g_calls, cache.h_calls,
        termination_reason
    )
    
    display_final(display, result)

    return result
end