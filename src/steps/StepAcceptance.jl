"""
Trust-region step acceptance and radius update logic.
"""

# Compute predicted reduction for trust-region model
function predicted_reduction(g::AbstractVector{T}, p::AbstractVector{T}, 
                           Hp::AbstractVector{T}) where {T<:Real}
    # Predicted reduction: pred = -g^T * p - 0.5 * p^T * H * p
    # Hp should contain H * p
    return -dot(g, p) - 0.5 * dot(p, Hp)
end

# Compute actual reduction
function actual_reduction(f_current::T, f_trial::T) where {T<:Real}
    return f_current - f_trial
end

# Trust-region acceptance test
function accept_step(rho::T, eta_1::T = T(0.25)) where {T<:Real}
    return rho > eta_1
end

# Update trust-region radius (following fides: Coleman & Li 1994/1996)
function update_trust_region_radius(Delta::T, rho::T, step_norm::T,
                                   mu::T = T(0.25), eta::T = T(0.75),
                                   gamma1::T = T(0.25), gamma2::T = T(2.0),
                                   max_Delta::T = T(1000.0)) where {T<:Real}
    if rho < mu
        # Poor model agreement — shrink radius
        Delta_new = gamma1 * Delta
    elseif rho > eta && step_norm >= T(0.9) * Delta
        # Good model agreement and step hit boundary — expand
        Delta_new = min(gamma2 * Delta, max_Delta)
    else
        # Maintain current radius
        Delta_new = Delta
    end
    
    return Delta_new
end

# Check convergence criteria
function check_convergence(g::AbstractVector{T}, p::AbstractVector{T}, 
                         f_rel_change::T, options) where {T<:Real}
    g_norm = norm(g)
    p_norm = norm(p)
    
    # Gradient tolerance
    if g_norm < options.gtol_a
        return true, :gtol
    end
    
    # Step tolerance (only if xtol is enabled and step has been computed)
    if options.xtol > zero(T) && p_norm > zero(T) && p_norm < options.xtol
        return true, :xtol
    end
    
    # Function tolerance (only check if we've actually taken a step)
    # f_rel_change being exactly zero means no step has been taken yet
    if options.ftol_a > zero(T) && !iszero(f_rel_change) && abs(f_rel_change) < options.ftol_a
        return true, :ftol
    end
    
    return false, :continue
end

# Fallback step computation (Cauchy step)
function compute_cauchy_step!(p::AbstractVector{T}, g::AbstractVector{T},
                            H_or_approx, cache::RetroCache{T}, Delta::T) where {T<:Real}
    g_norm = norm(g)
    
    if g_norm < eps(T)
        fill!(p, zero(T))
        return zero(T)
    end
    
    # Compute H * g for curvature along gradient direction
    try
        @. cache.tmp = g  # Fallback: use gradient as proxy for H*g
        
        gHg = dot(g, cache.tmp)
        
        if gHg > eps(T)
            alpha = min(g_norm^2 / gHg, Delta / g_norm)
        else
            alpha = Delta / g_norm
        end
        
        @. p = -alpha * g
        return alpha * g_norm
        
    catch
        # Ultimate fallback: steepest descent to boundary
        alpha = Delta / g_norm
        @. p = -alpha * g
        return Delta
    end
end