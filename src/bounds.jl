# ============================================================================
# Bound Constraint Utilities  
# ============================================================================

function project_bounds(x::AbstractVector, lb, ub)
    x_proj = copy(x)
    if lb !== nothing
        x_proj .= max.(x_proj, lb)
    end
    if ub !== nothing  
        x_proj .= min.(x_proj, ub)
    end
    return x_proj
end

function update_active_set!(state::TrustRegionState{T}, options) where T
    x, g = state.x, state.gx
    lb, ub = state.lb, state.ub
    
    fill!(state.active_set, false)
    
    # Mark variables active if at bounds with appropriate gradient sign
    for i in eachindex(x)
        if lb !== nothing && x[i] ≈ lb[i] && g[i] > 0
            state.active_set[i] = true
        elseif ub !== nothing && x[i] ≈ ub[i] && g[i] < 0
            state.active_set[i] = true
        end
    end
    
    # Compute free gradient (zero for active variables)
    state.gx_free .= state.gx
    state.gx_free[state.active_set] .= zero(T)
end

function apply_reflective_bounds!(state::TrustRegionState, options)
    step = state.step
    x = state.x
    lb, ub = state.lb, state.ub
    
    state.step_reflected .= step
    
    if lb === nothing && ub === nothing
        return  # No bounds
    end
    
    # Apply reflective bounds using the method from fides
    for i in eachindex(step)
        x_new = x[i] + step[i]
        
        if lb !== nothing && x_new < lb[i]
            # Reflect off lower bound
            dist_to_bound = x[i] - lb[i]
            if dist_to_bound > options.theta1 * state.tr_radius
                # Simple reflection
                state.step_reflected[i] = 2 * (lb[i] - x[i]) - step[i]
            else
                # Move to bound
                state.step_reflected[i] = lb[i] - x[i]
            end
        elseif ub !== nothing && x_new > ub[i]
            # Reflect off upper bound  
            dist_to_bound = ub[i] - x[i]
            if dist_to_bound > options.theta1 * state.tr_radius
                # Simple reflection
                state.step_reflected[i] = 2 * (ub[i] - x[i]) - step[i]
            else
                # Move to bound
                state.step_reflected[i] = ub[i] - x[i]
            end
        end
    end
    
    # Ensure reflected step is within trust region
    step_norm = norm(state.step_reflected)
    if step_norm > state.tr_radius
        state.step_reflected .*= state.tr_radius / step_norm
    end
end