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

# Main trust-region step computation
function compute_trust_region_step!(cache::RetroCache{T}, prob::RetroProblem, 
                                  subspace, subspace_state, hess_approx, hess_state,
                                  tr_solver, x::AbstractVector{T}, Delta::T,
                                  options) where {T<:Real}
    n = length(x)
    has_bounds = any(isfinite, prob.lb) || any(isfinite, prob.ub)
    step_norm = zero(T)

    if has_bounds
        # ── Coleman-Li affine scaling ────────────────────────────────
        # D = sqrt(|v|),  dv = Jacobian diagonal of v
        compute_affine_scaling!(cache.scaling, cache.d, x, cache.g, prob.lb, prob.ub)

        # Save originals (restored in `finally`)
        B_save = copy(cache.B)
        g_save = copy(cache.g)

        try
            # Scaled Hessian: B̂[i,j] = D[i]*B[i,j]*D[j] + δ_{ij}*|g[i]|*dv[i]
            for j in 1:n, i in 1:n
                cache.B[i,j] = cache.scaling[i] * B_save[i,j] * cache.scaling[j]
            end
            for i in 1:n
                cache.B[i,i] += abs(g_save[i]) * cache.d[i]
            end

            # Scaled gradient: sg = D * g
            @. cache.g = cache.scaling * g_save

            # Build subspace & solve TR in scaled space
            build_subspace!(subspace, subspace_state, cache, hess_approx, hess_state, x)
            step_norm = solve_subspace_tr!(tr_solver, subspace, subspace_state, cache, Delta)

            # Convert step from scaled to original space: s = D * ss
            @. cache.p *= cache.scaling

        catch e
            @warn "Subspace TR solve failed, using Cauchy step: $e"
            # Fallback: steepest descent in original space
            g_norm = norm(g_save)
            if g_norm > eps(T)
                alpha = Delta / g_norm
                @. cache.p = -alpha * g_save
            else
                fill!(cache.p, zero(T))
            end
        finally
            # Restore original Hessian and gradient
            copy!(cache.B, B_save)
            copy!(cache.g, g_save)
        end

        # Apply reflective bounds (Coleman & Li)
        apply_reflective_bounds!(cache.x_trial, x, cache.p, prob.lb, prob.ub, options.theta2;
                                g=cache.g)

        # Update step to the actual movement (post-reflection)
        @. cache.p = cache.x_trial - x

        # Step norm in scaled space (consistent with Delta)
        step_norm = zero(T)
        for i in 1:n
            ss_i = cache.p[i] / max(cache.scaling[i], eps(T))
            step_norm += ss_i * ss_i
        end
        step_norm = sqrt(step_norm)

    else
        # ── No bounds: simple unscaled path ──────────────────────────
        try
            build_subspace!(subspace, subspace_state, cache, hess_approx, hess_state, x)
            step_norm = solve_subspace_tr!(tr_solver, subspace, subspace_state, cache, Delta)
        catch e
            @warn "Subspace TR solve failed, using Cauchy step: $e"
            step_norm = compute_cauchy_step!(cache.p, cache.g, hess_approx, cache, Delta)
        end

        @. cache.x_trial = x + cache.p
        step_norm = norm(cache.p)
    end

    return step_norm
end

# Compute the Hessian-vector product for predicted reduction
function compute_hv_product!(Hp::AbstractVector{T}, hess_approx, hess_state, 
                           cache::RetroCache{T}, p::AbstractVector{T}) where {T<:Real}
    try
        apply_hessian!(Hp, hess_approx, hess_state, cache, p)
    catch e
        @warn "Hessian-vector product failed, using identity: $e"
        copy!(Hp, p)
    end
end

# Check for negative curvature and handle accordingly
function check_negative_curvature(g::AbstractVector{T}, p::AbstractVector{T}, 
                                Hp::AbstractVector{T}, Delta::T) where {T<:Real}
    pHp = dot(p, Hp)
    
    if pHp <= zero(T)
        # Negative curvature detected
        g_norm = norm(g)
        if g_norm > eps(T)
            # Use steepest descent to boundary
            alpha = Delta / g_norm
            @. p = -alpha * g
            return true, alpha * g_norm
        else
            # At critical point
            fill!(p, zero(T))
            return true, zero(T)
        end
    end
    
    return false, norm(p)
end

# Model quality assessment
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