"""
Trust-region step acceptance and radius update logic.
"""

function predicted_reduction(g::AbstractVector{T}, p::AbstractVector{T}, 
                           Hp::AbstractVector{T}) where {T<:Real}
    return -dot(g, p) - 0.5 * dot(p, Hp)
end

function actual_reduction(f_current::T, f_trial::T) where {T<:Real}
    return f_current - f_trial
end

function accept_step(rho::T, eta_1::T = T(0.25)) where {T<:Real}
    return rho > eta_1
end

function update_trust_region_radius(Delta::T, rho::T, step_norm::T,
                                   mu::T = T(0.25), eta::T = T(0.75),
                                   gamma1::T = T(0.25), gamma2::T = T(2.0),
                                   max_Delta::T = T(1000.0)) where {T<:Real}
    if rho < mu
        Delta_new = gamma1 * Delta
    elseif rho > eta && step_norm >= T(0.9) * Delta
        Delta_new = min(gamma2 * Delta, max_Delta)
    else
        Delta_new = Delta
    end
    
    return Delta_new
end

function check_convergence(g::AbstractVector{T}, p::AbstractVector{T}, 
                         f_rel_change::T, options) where {T<:Real}
    g_norm = norm(g)
    p_norm = norm(p)
    
    if g_norm < options.gtol_a
        return true, :gtol
    end
    
    if options.xtol > zero(T) && p_norm > zero(T) && p_norm < options.xtol
        return true, :xtol
    end
    
    if options.ftol_a > zero(T) && !iszero(f_rel_change) && abs(f_rel_change) < options.ftol_a
        return true, :ftol
    end
    
    return false, :continue
end

function compute_cauchy_step!(p::AbstractVector{T}, g::AbstractVector{T}, cache::RetroCache{T}, Delta::T) where {T<:Real}
    g_norm = norm(g)
    
    if g_norm < eps(T)
        fill!(p, zero(T))
        return zero(T)
    end
    
    try
        @. cache.tmp = g 
        
        gHg = dot(g, cache.tmp)
        
        if gHg > eps(T)
            alpha = min(g_norm^2 / gHg, Delta / g_norm)
        else
            alpha = Delta / g_norm
        end
        
        @. p = -alpha * g
        return alpha * g_norm
        
    catch
        alpha = Delta / g_norm
        @. p = -alpha * g
        return Delta
    end
end