"""
Trust-region step acceptance and radius update logic.
"""

function predicted_reduction(g::AbstractVector{T}, p::AbstractVector{T}, 
                           Hp::AbstractVector{T}) where {T<:Real}
    return -dot(g, p) - 0.5 * dot(p, Hp)
end

function predicted_reduction_bounded(
    g::AbstractVector{T},
    p::AbstractVector{T},
    Hp::AbstractVector{T},
    dv::AbstractVector{T},
    scaling::AbstractVector{T},
) where {T<:Real}
    diag_term = zero(T)
    @inbounds for i in eachindex(p, g, dv, scaling)
        ss_i = p[i] / max(scaling[i], eps(T))
        diag_term += abs(g[i]) * dv[i] * ss_i * ss_i
    end
    return -dot(g, p) - T(0.5) * (dot(p, Hp) + diag_term)
end

function actual_reduction(
    f_current::T,
    f_trial::T,
    stepsx::AbstractVector{T},
    grad_trial::AbstractVector{T},
    dv::AbstractVector{T},
    scaling::AbstractVector{T},
) where {T<:Real}
    aug = zero(T)
    @inbounds for i in eachindex(stepsx, grad_trial, dv, scaling)
        ss_i = stepsx[i] / max(scaling[i], eps(T))
        aug += ss_i * abs(grad_trial[i]) * dv[i] * ss_i
    end
    aug *= T(0.5)
    return f_current - f_trial - aug
end

function accept_step(rho::T, eta_1::T = T(0.25)) where {T<:Real}
    return rho > 0.0
end

function update_trust_region_radius(Delta::T, rho::T, step_norm::T,
                                   mu::T = T(0.25), eta::T = T(0.75),
                                   gamma1::T = T(0.25), gamma2::T = T(2.0),
                                   max_Delta::T = T(1000.0)) where {T<:Real}

    if rho <= mu
        # Avoid collapsing Delta to zero when step_norm is numerically zero.
        step_term = step_norm > eps(T) ? (step_norm / T(4.0)) : (gamma1 * Delta)
        Delta_new = min(gamma1 * Delta, step_term)
    elseif rho >= eta && step_norm >= T(0.9) * Delta
        Delta_new = min(gamma2 * Delta, max_Delta)
    else
        Delta_new = Delta
    end
    
    return Delta_new
end

function check_convergence(g::AbstractVector{T}, p::AbstractVector{T}, 
                         f_change::T, f_current::T, options) where {T<:Real}
    g_norm = norm(g)
    p_norm = norm(p)
    
    if g_norm < options.gtol_a
        return true, :gtol
    end
    
    if options.xtol > zero(T) && p_norm > zero(T) && p_norm < options.xtol
        return true, :xtol
    end
    
    if options.ftol_a > zero(T) && !iszero(f_change) && abs(f_change) < options.ftol_a
        return true, :ftol
    end

    if options.ftol_r > zero(T) && !iszero(f_change)
        rel_change = abs(f_change) / max(abs(f_current), one(T))
        if rel_change < options.ftol_r
            return true, :ftol
        end
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