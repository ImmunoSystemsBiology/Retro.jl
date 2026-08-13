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

function _load_exact_hessian_matrix!(
    cache::RetroCache{T},
    hess_approx,
    hess_state,
    n::Int,
) where {T<:Real}
    if !(hess_approx isa ExactHessian)
        return
    end

    if hasproperty(hess_state, :H) && size(getproperty(hess_state, :H)) == size(cache.B)
        copy!(cache.B, getproperty(hess_state, :H))
        return
    end

    # Fallback for unexpected state shape.
    for i in 1:n
        fill!(cache.tmp, zero(T))
        cache.tmp[i] = one(T)
        apply_hessian!(cache.v1, hess_approx, hess_state, cache, cache.tmp)
        @inbounds for j in 1:n
            cache.B[j, i] = cache.v1[j]
        end
    end
end

function _dynamic_stepback_theta(
    cache::RetroCache{T},
    theta_max::T,
    n::Int,
) where {T<:Real}
    vg_inf = zero(T)
    @inbounds for i in 1:n
        v_abs_i = cache.scaling[i] * cache.scaling[i]
        vg_inf = max(vg_inf, abs(v_abs_i * cache.g[i]))
    end
    return clamp(max(theta_max, one(T) - vg_inf), theta_max, one(T))
end

function _model_q!(
    cache::RetroCache{T},
    x::AbstractVector{T},
    xcand::AbstractVector{T},
    n::Int,
) where {T<:Real}
    @inbounds for i in 1:n
        cache.tmp[i] = (xcand[i] - x[i]) / max(cache.scaling[i], eps(T))
    end

    @inbounds for i in 1:n
        cache.v2[i] = cache.scaling[i] * cache.tmp[i]
    end

    mul!(cache.v1, cache.B_save, cache.v2)

    @inbounds for i in 1:n
        cache.v1[i] = cache.scaling[i] * cache.v1[i] + abs(cache.g_save[i]) * cache.d[i] * cache.tmp[i]
    end

    q_lin = zero(T)
    @inbounds for i in 1:n
        q_lin += (cache.scaling[i] * cache.g_save[i]) * cache.tmp[i]
    end

    return q_lin + T(0.5) * dot(cache.tmp, cache.v1)
end

function _consider_candidate!(
    cache::RetroCache{T},
    x::AbstractVector{T},
    xcand::AbstractVector{T},
    n::Int,
    q_best::T,
) where {T<:Real}
    q_cand = _model_q!(cache, x, xcand, n)
    if q_cand < q_best
        copy!(cache.x_trial, xcand)
        return q_cand
    end
    return q_best
end

function _select_bounded_candidate!(
    cache::RetroCache{T},
    prob::RetroProblem,
    x::AbstractVector{T},
    Delta::T,
    theta_stepback::T,
    n::Int,
) where {T<:Real}
    x_work = cache.x_work
    x_trunc = cache.x_trunc
    x_cauchy = cache.x_cauchy

    q_best = typemax(T)

    α_bound0, _, hit_bound0 = find_step_to_bound(x, cache.p, prob.lb, prob.ub)
    if hit_bound0 == :none
        @. x_work = x + cache.p
    else
        α_step0 = min(one(T), theta_stepback * α_bound0) * (one(T) - T(1e-10))
        @. x_work = x + α_step0 * cache.p
        project_bounds!(x_work, prob.lb, prob.ub)
    end
    q_best = _consider_candidate!(cache, x, x_work, n, q_best)

    max_reflections = max(n - 1, 1)
    for nref in 1:max_reflections
        apply_reflective_bounds!(
            x_work,
            x,
            cache.p,
            prob.lb,
            prob.ub,
            theta_stepback;
            g = cache.g,
            max_reflections = nref,
        )
        q_best = _consider_candidate!(cache, x, x_work, n, q_best)
    end

    copy!(x_trunc, x)
    copy!(cache.v2, cache.p)
    for _ in 1:n
        α_bound, hit_index, hit_bound = find_step_to_bound(x_trunc, cache.v2, prob.lb, prob.ub)

        if hit_bound == :none
            @. x_trunc = x_trunc + cache.v2
            project_bounds!(x_trunc, prob.lb, prob.ub)
            q_best = _consider_candidate!(cache, x, x_trunc, n, q_best)
            break
        end

        α_step = min(one(T), theta_stepback * α_bound) * (one(T) - T(1e-10))
        @. x_trunc = x_trunc + α_step * cache.v2
        project_bounds!(x_trunc, prob.lb, prob.ub)
        q_best = _consider_candidate!(cache, x, x_trunc, n, q_best)

        ss0_norm2 = zero(T)
        @inbounds for i in 1:n
            ss0_i = (x_trunc[i] - x[i]) / max(cache.scaling[i], eps(T))
            ss0_norm2 += ss0_i * ss0_i
        end
        Δ_rem = sqrt(max(Delta * Delta - ss0_norm2, zero(T)))
        if Δ_rem <= eps(T) || hit_index == 0
            break
        end

        cache.v2[hit_index] = zero(T)

        ssdir_norm2 = zero(T)
        @inbounds for i in 1:n
            ssdir_i = cache.v2[i] / max(cache.scaling[i], eps(T))
            ssdir_norm2 += ssdir_i * ssdir_i
        end
        ssdir_norm = sqrt(ssdir_norm2)
        if ssdir_norm <= eps(T)
            break
        end

        α_free = min(one(T), Δ_rem / ssdir_norm)
        @. cache.v2 = α_free * cache.v2
    end

    compute_cauchy_boundary_point!(x_cauchy, x, cache.g, prob.lb, prob.ub, Delta)
    q_best = _consider_candidate!(cache, x, x_cauchy, n, q_best)

    return q_best
end

function compute_trust_region_step!(cache::RetroCache{T}, prob::RetroProblem, 
                                  subspace, subspace_state, hess_approx, hess_state,
                                  tr_solver, x::AbstractVector{T}, Delta::T,
                                  options) where {T<:Real}
    n = length(x)
    has_bounds = any(isfinite, prob.lb) || any(isfinite, prob.ub)
    step_norm = zero(T)
    cache.pred_red_model = zero(T)

    _load_exact_hessian_matrix!(cache, hess_approx, hess_state, n)

    if has_bounds
        compute_affine_scaling!(cache.scaling, cache.d, x, cache.g, prob.lb, prob.ub)

        theta_stepback = _dynamic_stepback_theta(cache, options.theta_max, n)

        copy!(cache.B_save, cache.B)
        copy!(cache.g_save, cache.g)

        try
            for j in 1:n, i in 1:n
                cache.B[i,j] = cache.scaling[i] * cache.B_save[i,j] * cache.scaling[j]
            end
            for i in 1:n
                cache.B[i,i] += abs(cache.g_save[i]) * cache.d[i]
            end

            @. cache.g = cache.scaling * cache.g_save

            build_subspace!(subspace, subspace_state, cache, hess_approx, hess_state, x, Delta)
            step_norm = solve_subspace_tr!(tr_solver, subspace, subspace_state, cache, Delta)

            @. cache.p *= cache.scaling

        catch e
            @warn "Subspace TR solve failed, using Cauchy step: $e"
            g_norm = norm(cache.g_save)
            if g_norm > eps(T)
                alpha = Delta / g_norm
                @. cache.p = -alpha * cache.g_save
            else
                fill!(cache.p, zero(T))
            end
        finally
            copy!(cache.B, cache.B_save)
            copy!(cache.g, cache.g_save)
        end

        q_best = _select_bounded_candidate!(cache, prob, x, Delta, theta_stepback, n)

        # Keep the predicted decrease from the same transformed model used
        # for candidate selection so acceptance/radius updates use consistent
        # trust-region ratio semantics.
        cache.pred_red_model = -q_best

        @. cache.p = cache.x_trial - x

        step_norm = zero(T)
        @inbounds for i in 1:n
            ss_i = cache.p[i] / max(cache.scaling[i], eps(T))
            step_norm += ss_i * ss_i
        end
        step_norm = sqrt(step_norm)

    else
        try
            build_subspace!(subspace, subspace_state, cache, hess_approx, hess_state, x, Delta)
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