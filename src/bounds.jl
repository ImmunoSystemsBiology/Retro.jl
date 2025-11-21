"""
    Bound Constraint Handling

Functions for managing bound constraints in trust-region optimization using the
Coleman-Li reflective barrier approach.
"""

"""
    compute_affine_scaling!(state)

Compute affine scaling vectors v and dv according to Coleman-Li methodology.

For variables constrained by bounds, v measures the distance to the active bound.
For unconstrained variables, v = sign(gradient). The derivative dv is used in
the scaled Hessian approximation.

This scaling transforms the constrained problem into one more suitable for
trust-region methods near boundaries.

Reference: Coleman & Li (1994), "On the Convergence of Interior-Reflective Newton Methods
for Nonlinear Minimization Subject to Bounds"
"""
function compute_affine_scaling!(state::TrustRegionState{T}) where T
    x = state.x
    g = state.grad
    lb, ub = state.lb, state.ub
    v = state.v
    dv = state.dv
    
    # Fused loop: compute both default and bounded cases in single pass
    @inbounds for i in eachindex(x)
        gi = g[i]
        xi = x[i]
        
        # Determine which bound is active based on gradient direction
        if gi < 0 && isfinite(ub[i])
            # Upper bound is relevant (we're moving towards it)
            v[i] = xi - ub[i]
            dv[i] = one(T)
        elseif gi >= 0 && isfinite(lb[i])
            # Lower bound is relevant
            v[i] = xi - lb[i]
            dv[i] = one(T)
        else
            # Default: sign scaling for unconstrained variables
            v[i] = gi != zero(T) ? sign(gi) : one(T)
            dv[i] = zero(T)
        end
    end
end

"""
    check_and_project_bounds!(x, lb, ub)

    Check if `x` is within bounds defined by `lb` and `ub`. If not, project `x` onto the feasible region.
Project a point `x` in place onto the feasible region defined by bounds `lb` and `ub`.

# Arguments
- `x`: Point to project
- `lb`: Lower bounds (can be nothing)
- `ub`: Upper bounds (can be nothing)

"""
function check_and_project_bounds!(x::AbstractVector{T}, lb::AbstractVector{T}, ub::AbstractVector{T}) where T

    if any(lb .> x) || any(ub .< x)
        @warn "Initial point is out of set bounds. Projecting onto feasible region."
    end

    x .= max.(x, lb)
    x .= min.(x, ub)
end

"""
    update_active_set!(state, options)

Identify which bound constraints are currently active.

A constraint is active if the variable is at the bound and the gradient
points in the direction that would violate the bound.

Updates `state.active_set` and `state.gx_free` (gradient with active components zeroed).
"""
function update_active_set!(state::TrustRegionState{T}, options) where T
    x = state.x 
    g = state.grad
    gx_free = state.gx_free
    lb, ub = state.lb, state.ub
    active_set = state.active_set
    bound_tol = sqrt(eps(T))
    
    # Fused loop: update active set and gx_free in single pass
    @inbounds for i in eachindex(x)
        xi = x[i]
        gi = g[i]
        
        # Variable is active if it's at a bound and gradient points outward
        at_lower = isfinite(lb[i]) && abs(xi - lb[i]) < bound_tol
        at_upper = isfinite(ub[i]) && abs(xi - ub[i]) < bound_tol
        
        is_active = (at_lower && gi > 0) || (at_upper && gi < 0)
        active_set[i] = is_active
        gx_free[i] = is_active ? zero(T) : gi
    end
end

"""
    apply_reflective_bounds!(state, options)

Apply reflective boundary conditions to the proposed step using Coleman-Li method.

When a proposed step would violate bounds, this function applies reflection or
truncation based on the distance to the boundary controlled by theta parameters.

The step is modified such that it respects bounds while still making progress.
This implements the reflective step-back strategy from Coleman & Li (1994, 1996).

Updates `state.step_reflected` with the step after bounds handling.
"""
function apply_reflective_bounds!(state::TrustRegionState{T}, options) where T
    step = state.step
    x = state.x
    lb, ub = state.lb, state.ub
    step_reflected = state.step_reflected
    
    # Start with the proposed step
    @. step_reflected = step
    
    # If no bounds, nothing to do
    if all(.!isfinite.(lb)) && all(.!isfinite.(ub))
        return
    end
    
    # Compute distance to boundaries for each component
    minbr = T(Inf)
    step_tol = eps(T)
    
    @inbounds for i in eachindex(step)
        si = step[i]
        if abs(si) > step_tol
            xi = x[i]
            # Distance to boundary as fraction of step
            dist_lower = isfinite(lb[i]) ? (lb[i] - xi) / si : T(Inf)
            dist_upper = isfinite(ub[i]) ? (ub[i] - xi) / si : T(Inf)
            
            # Maximum positive fraction we can take
            br_i = max(dist_lower, dist_upper)
            if br_i < minbr && br_i > 0
                minbr = br_i
            end
        end
    end
    
    # Apply step-back if we hit a boundary
    if minbr < 1
        alpha = min(one(T), options.theta1 * minbr)
        @. step_reflected *= alpha
    end
    
    # Ensure step doesn't violate bounds (safety clamp)
    @inbounds for i in eachindex(step_reflected)
        x_new = x[i] + step_reflected[i]
        
        if isfinite(lb[i]) && x_new < lb[i]
            step_reflected[i] = lb[i] - x[i]
        elseif isfinite(ub[i]) && x_new > ub[i]
            step_reflected[i] = ub[i] - x[i]
        end
    end
end