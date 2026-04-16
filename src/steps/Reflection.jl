"""
Reflective bounds handling for box-constrained optimization.

Following Coleman & Li (1994, 1996), this implements the reflective step-back strategy
with support for multiple reflections (similar to fides). When a step hits a boundary,
it can be reflected back into the feasible region. The algorithm continues reflecting
until:
1. No boundary is hit (step stays interior)
2. A local minimum along the reflected path is encountered
3. Maximum reflections is reached (safety limit)

The final step is selected among:
1. The reflection path (if it improves the model)
2. The constrained Cauchy point (steepest descent to boundary)
3. The truncated step at the boundary

Implementation follows fides: allows arbitrarily many reflections until the first
local minimum. In contrast, fmincon/lsqnonlin/ls_trf allow only a single reflection.
"""

# Default parameters
const DEFAULT_BOUNDS_EPSILON = 1e-8  # Minimum distance from bounds
const DEFAULT_MAX_REFLECTIONS = 10   # Maximum number of reflections

"""
    initialize_away_from_bounds!(x, lb, ub; epsilon=1e-8)

Move initial point away from bounds if too close or exactly on boundary.
This prevents degeneracy in the bound-constrained scaling and ensures
the initial trust region step has room to explore.

For each variable:
- If x[i] ≤ lb[i] + ε, move to lb[i] + min(ε, (ub[i]-lb[i])/2)
- If x[i] ≥ ub[i] - ε, move to ub[i] - min(ε, (ub[i]-lb[i])/2)

Returns true if any modifications were made.
"""
function initialize_away_from_bounds!(x::AbstractVector{T}, lb::AbstractVector{T}, 
                                     ub::AbstractVector{T}; 
                                     epsilon::T = T(DEFAULT_BOUNDS_EPSILON)) where {T<:Real}
    modified = false
    n = length(x)
    
    for i in 1:n
        has_lb = isfinite(lb[i])
        has_ub = isfinite(ub[i])
        
        if has_lb && has_ub
            width = ub[i] - lb[i]
            rel_eps = min(epsilon, width * T(0.01))
            
            if x[i] <= lb[i] + rel_eps
                x[i] = lb[i] + rel_eps
                modified = true
            elseif x[i] >= ub[i] - rel_eps
                x[i] = ub[i] - rel_eps
                modified = true
            end
            
        elseif has_lb
            if x[i] <= lb[i] + epsilon
                x[i] = lb[i] + epsilon
                modified = true
            end
            
        elseif has_ub
            if x[i] >= ub[i] - epsilon
                x[i] = ub[i] - epsilon
                modified = true
            end
        end
    end
    
    return modified
end

"""
    compute_scaling!(scaling, x, lb, ub, theta1, theta2)

Compute the Coleman-Li scaling factors for bound-constrained optimization.
The scaling matrix D is diagonal with:
  D[i,i] = |v[i]|^{1/2}
where v[i] reflects distance to bounds in the gradient direction.

This scaling makes the trust-region ellipsoidal in a way that respects bounds.
"""
function compute_scaling!(scaling::AbstractVector{T}, x::AbstractVector{T}, 
                        lb::AbstractVector{T}, ub::AbstractVector{T}, 
                        theta1::T, theta2::T) where {T<:Real}
    n = length(x)
    
    for i in 1:n
        has_lb = isfinite(lb[i])
        has_ub = isfinite(ub[i])
        
        if has_lb && has_ub
            width = ub[i] - lb[i]
            dist_to_lb = (x[i] - lb[i]) / width
            dist_to_ub = (ub[i] - x[i]) / width
            
            min_dist = min(dist_to_lb, dist_to_ub)
            if min_dist < theta1
                scaling[i] = max(min_dist / theta1, T(0.01))
            else
                scaling[i] = one(T)
            end
            
        elseif has_lb
            dist = x[i] - lb[i]
            scale_dist = max(abs(x[i]), one(T)) * theta1
            if dist < scale_dist
                scaling[i] = max(dist / scale_dist, T(0.01))
            else
                scaling[i] = one(T)
            end
            
        elseif has_ub
            dist = ub[i] - x[i]
            scale_dist = max(abs(x[i]), one(T)) * theta1
            if dist < scale_dist
                scaling[i] = max(dist / scale_dist, T(0.01))
            else
                scaling[i] = one(T)
            end
            
        else
            scaling[i] = one(T)
        end
    end
end

"""
    scale_gradient!(scaled_g, g, scaling)

Apply scaling to gradient: scaled_g = D * g
"""
function scale_gradient!(scaled_g::AbstractVector{T}, g::AbstractVector{T}, 
                        scaling::AbstractVector{T}) where {T<:Real}
    @. scaled_g = g * scaling
end

"""
    compute_affine_scaling!(D, dv, x, g, lb, ub)

Compute the Coleman-Li affine scaling (Definition 2, Coleman & Li 1994).

For each variable:
- If the gradient points toward a finite bound: `v = x - bound`, `dv = 1`
- Otherwise (unconstrained direction): `v = sign(g)`, `dv = 0`

The relevant bound depends on gradient sign: `lb` if `g ≥ 0`, `ub` if `g < 0`.

Stores `D = √|v|` (clamped away from zero) and the Jacobian diagonal `dv`.
"""
function compute_affine_scaling!(D::AbstractVector{T}, dv::AbstractVector{T},
                                 x::AbstractVector{T}, g::AbstractVector{T},
                                 lb::AbstractVector{T}, ub::AbstractVector{T}) where {T<:Real}
    for i in eachindex(x)
        if g[i] >= zero(T) && isfinite(lb[i])
            v_i = x[i] - lb[i]
            dv[i] = one(T)
        elseif g[i] < zero(T) && isfinite(ub[i])
            v_i = x[i] - ub[i]
            dv[i] = one(T)
        else
            v_i = g[i] != zero(T) ? sign(g[i]) : one(T)
            dv[i] = zero(T)
        end

        D[i] = max(sqrt(abs(v_i)), T(1e-10))
    end
end

"""
    find_step_to_bound(x, p, lb, ub)

Find the maximum step length α ∈ (0, 1] such that x + α*p stays feasible.
Returns (α, hit_index, hit_bound) where:
- α: maximum feasible step length
- hit_index: index of first bound hit (0 if none)
- hit_bound: :lower, :upper, or :none
"""
function find_step_to_bound(x::AbstractVector{T}, p::AbstractVector{T},
                           lb::AbstractVector{T}, ub::AbstractVector{T}) where {T<:Real}
    α_max = one(T)
    hit_index = 0
    hit_bound = :none
    
    for i in eachindex(x)
        if p[i] < -eps(T) && isfinite(lb[i])
            α_i = (lb[i] - x[i]) / p[i]
            if α_i > zero(T) && α_i < α_max
                α_max = α_i
                hit_index = i
                hit_bound = :lower
            end
        elseif p[i] > eps(T) && isfinite(ub[i])
            α_i = (ub[i] - x[i]) / p[i]
            if α_i > zero(T) && α_i < α_max
                α_max = α_i
                hit_index = i
                hit_bound = :upper
            end
        end
    end
    
    return α_max, hit_index, hit_bound
end

"""
    reflect_step!(p, hit_index, hit_bound, g, H_diag)

Reflect the step at the boundary. The reflected direction is computed
following Coleman & Li: the component hitting the boundary is negated,
while other components are adjusted based on local curvature.

If gradient at boundary indicates a local minimum in that direction,
the reflection is not performed (return false).
"""
function reflect_step!(p::AbstractVector{T}, hit_index::Int, hit_bound::Symbol,
                      g::AbstractVector{T}) where {T<:Real}
    if hit_bound == :lower && g[hit_index] > zero(T)
        return false  
    elseif hit_bound == :upper && g[hit_index] < zero(T)
        return false 
    end

    p[hit_index] = -p[hit_index]
    
    return true
end

"""
    apply_reflective_bounds!(x_trial, x, p, lb, ub, theta2; 
                            g=nothing, max_reflections=10)

Apply multiple reflective bounds following Coleman & Li (1994, 1996) and fides.

The algorithm:
1. Take step until hitting a boundary
2. Reflect off the boundary (negate component)
3. Repeat until no boundary is hit, local minimum found, or max_reflections reached
4. Apply safety clamping to ensure strict feasibility

When gradient `g` is provided, the algorithm checks for local minima at boundaries
and stops reflecting if the boundary is optimal in that direction.
"""
function apply_reflective_bounds!(x_trial::AbstractVector{T}, x::AbstractVector{T},
                                p::AbstractVector{T}, lb::AbstractVector{T}, 
                                ub::AbstractVector{T}, theta2::T;
                                g::Union{Nothing, AbstractVector{T}} = nothing,
                                max_reflections::Int = DEFAULT_MAX_REFLECTIONS) where {T<:Real}
    n = length(x)
    
    @. x_trial = x
    
    p_remaining = copy(p)
    
    for _ in 1:max_reflections
        α_to_bound, hit_index, hit_bound = find_step_to_bound(x_trial, p_remaining, lb, ub)
        
        if hit_bound == :none || α_to_bound >= one(T) - eps(T)
            @. x_trial = x_trial + p_remaining
            break
        end
        
        α_step = α_to_bound * (one(T) - T(1e-10)) 
        @. x_trial = x_trial + α_step * p_remaining
        
        α_remaining = one(T) - α_step
        @. p_remaining = α_remaining * p_remaining
        
        should_reflect = true
        if g !== nothing
            if hit_bound == :lower && g[hit_index] > zero(T)
                should_reflect = false 
            elseif hit_bound == :upper && g[hit_index] < zero(T)
                should_reflect = false  
            end
        end
        
        if !should_reflect
            p_remaining[hit_index] = zero(T)
            
            # If nothing left, stop
            if norm(p_remaining) < eps(T) * norm(p)
                break
            end
            continue  
        end
        
        p_remaining[hit_index] = -p_remaining[hit_index]

        if norm(p_remaining) < eps(T) * norm(p)
            break
        end
    end
    
    eps_interior = T(DEFAULT_BOUNDS_EPSILON)
    for i in 1:n
        if isfinite(lb[i])
            x_trial[i] = max(x_trial[i], lb[i] + eps_interior)
        end
        if isfinite(ub[i])
            x_trial[i] = min(x_trial[i], ub[i] - eps_interior)
        end
    end
end

"""
    project_bounds!(x, lb, ub)

Simple projection onto bounds (hard clamping).
"""
function project_bounds!(x::AbstractVector{T}, lb::AbstractVector{T}, 
                        ub::AbstractVector{T}) where {T<:Real}
    for i in eachindex(x)
        if isfinite(lb[i])
            x[i] = max(x[i], lb[i])
        end
        if isfinite(ub[i])
            x[i] = min(x[i], ub[i])
        end
    end
end

"""
    projected_gradient_norm(g, x, lb, ub; tol=1e-8)

Compute the norm of the projected (free) gradient for bound-constrained
optimality.  At a constrained optimum every free component of the gradient
is zero; the remaining (active-bound) components are allowed to be nonzero
because they point into the infeasible region.

A gradient component is considered "active" (and zeroed out) when the
variable sits at the bound (within `tol`) and the gradient would push it
further into the infeasible region:

* ``x_i \\le \\mathrm{lb}_i + \\mathrm{tol}`` **and** ``g_i > 0``
* ``x_i \\ge \\mathrm{ub}_i - \\mathrm{tol}`` **and** ``g_i < 0``
"""
function projected_gradient_norm(g::AbstractVector{T}, x::AbstractVector{T},
                                 lb::AbstractVector{T}, ub::AbstractVector{T};
                                 tol::T = T(1e-8)) where {T<:Real}
    s = zero(T)
    @inbounds for i in eachindex(g)
        gi = g[i]
        at_lb = isfinite(lb[i]) && x[i] <= lb[i] + tol
        at_ub = isfinite(ub[i]) && x[i] >= ub[i] - tol
        if (at_lb && gi > zero(T)) || (at_ub && gi < zero(T))
            continue
        end
        s += gi * gi
    end
    return sqrt(s)
end

"""
    compute_cauchy_boundary_point!(x_cauchy, x, g, lb, ub, Delta)

Compute the Cauchy point constrained to bounds: the point along -g direction
that minimizes a quadratic model within the trust region and bounds.

This provides a fallback when reflection fails to improve the model.
"""
function compute_cauchy_boundary_point!(x_cauchy::AbstractVector{T}, x::AbstractVector{T},
                                       g::AbstractVector{T}, lb::AbstractVector{T}, 
                                       ub::AbstractVector{T}, Delta::T) where {T<:Real}
    n = length(x)
    g_norm = norm(g)
    
    if g_norm < eps(T)
        @. x_cauchy = x
        return
    end

    descent_dir = -g / g_norm
    
    α_max = Delta 
    
    for i in 1:n
        if descent_dir[i] < -eps(T) && isfinite(lb[i])
            α_i = (lb[i] - x[i]) / descent_dir[i]
            if α_i > zero(T)
                α_max = min(α_max, α_i)
            end
        elseif descent_dir[i] > eps(T) && isfinite(ub[i])
            α_i = (ub[i] - x[i]) / descent_dir[i]
            if α_i > zero(T)
                α_max = min(α_max, α_i)
            end
        end
    end

    @. x_cauchy = x + α_max * descent_dir

    project_bounds!(x_cauchy, lb, ub)
end