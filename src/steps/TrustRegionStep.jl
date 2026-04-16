"""
Main trust-region step computation interface.
Coordinates between subspace methods, TR solvers, and bound constraints.

When bounds are present, uses Coleman-Li affine scaling (Eq 2.5,
Coleman & Li 1996): the Hessian is transformed to
`B_hat = D * B * D + diag(|g| .* dv)` and the gradient to `sg = D * g`,
where `D = diag(sqrt(|v|))`.
The TR subproblem is solved in this scaled space, then the step is
converted back: `s = D .* ss`.
"""

function compute_trust_region_step!(cache::RetroCache{T}, prob::RetroProblem, 
                                  subspace, subspace_state, hess_approx, hess_state,
                                  tr_solver, x::AbstractVector{T}, Delta::T,
                                  options) where {T<:Real}
    n = length(x)
    has_bounds = any(isfinite, prob.lb) || any(isfinite, prob.ub)
    step_norm = zero(T)

    if has_bounds
        compute_affine_scaling!(cache.scaling, cache.d, x, cache.g, prob.lb, prob.ub)

        B_save = copy(cache.B)
        g_save = copy(cache.g)

        try
            for j in 1:n, i in 1:n
                cache.B[i,j] = cache.scaling[i] * B_save[i,j] * cache.scaling[j]
            end
            for i in 1:n
                cache.B[i,i] += abs(g_save[i]) * cache.d[i]
            end

            @. cache.g = cache.scaling * g_save

            build_subspace!(subspace, subspace_state, cache, hess_approx, hess_state, x)
            step_norm = solve_subspace_tr!(tr_solver, subspace, subspace_state, cache, Delta)

            @. cache.p *= cache.scaling

        catch e
            @warn "Subspace TR solve failed, using Cauchy step: $e"
            g_norm = norm(g_save)
            if g_norm > eps(T)
                alpha = Delta / g_norm
                @. cache.p = -alpha * g_save
            else
                fill!(cache.p, zero(T))
            end
        finally
            copy!(cache.B, B_save)
            copy!(cache.g, g_save)
        end

        apply_reflective_bounds!(cache.x_trial, x, cache.p, prob.lb, prob.ub, options.theta2;
                                g=cache.g)

        @. cache.p = cache.x_trial - x

        step_norm = zero(T)
        for i in 1:n
            ss_i = cache.p[i] / max(cache.scaling[i], eps(T))
            step_norm += ss_i * ss_i
        end
        step_norm = sqrt(step_norm)

    else
        try
            build_subspace!(subspace, subspace_state, cache, hess_approx, hess_state, x)
            step_norm = solve_subspace_tr!(tr_solver, subspace, subspace_state, cache, Delta)
        catch e
            # TODO: I think this is redundant as all solvers have their own fallback. I'll leave it for now but we may want to remove it later.
            @warn "Subspace TR solve failed, using Cauchy step: $e"
            step_norm = compute_cauchy_step!(cache.p, cache.g, cache, Delta)
        end

        @. cache.x_trial = x + cache.p
        step_norm = norm(cache.p)
    end

    return step_norm
end

function compute_hv_product!(Hp::AbstractVector{T}, hess_approx, hess_state, 
                           cache::RetroCache{T}, p::AbstractVector{T}) where {T<:Real}
    try
        apply_hessian!(Hp, hess_approx, hess_state, cache, p)
    catch e
        @warn "Hessian-vector product failed, using identity: $e"
        copy!(Hp, p)
    end
end

function check_negative_curvature(g::AbstractVector{T}, p::AbstractVector{T}, 
                                Hp::AbstractVector{T}, Delta::T) where {T<:Real}
    pHp = dot(p, Hp)
    
    if pHp <= zero(T)
        g_norm = norm(g)
        if g_norm > eps(T)
            alpha = Delta / g_norm
            @. p = -alpha * g
            return true, alpha * g_norm
        else
            fill!(p, zero(T))
            return true, zero(T)
        end
    end
    
    return false, norm(p)
end

function assess_model_quality(rho::T) where {T<:Real}
    if rho < T(0.1)
        return :very_poor
    elseif rho < T(0.25)
        return :poor
    elseif rho < T(0.75)
        return :acceptable
    else
        return :good
    end
end