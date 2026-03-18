"""
    SR1 <: AbstractHessianApproximation

Symmetric Rank-1 quasi-Newton Hessian approximation.
Does not maintain positive definiteness but can handle indefinite Hessians.

# Fields
- `B0_scale::Real`: Initial Hessian scaling factor (default: 1.0)
- `skip_threshold::Real`: Skip update if denominator is too small
"""
struct SR1{T<:Real} <: AbstractHessianApproximation
    B0_scale::T
    skip_threshold::T
    
    SR1{T}(; B0_scale::T = one(T), skip_threshold::T = T(1e-8)) where {T} = new{T}(B0_scale, skip_threshold)
end

SR1(; kwargs...) = SR1{Float64}(; kwargs...)

# SR1 state
mutable struct SR1State{T<:Real}
    α::T  # Update coefficient
    initialized::Bool
    
    SR1State{T}() where {T} = new{T}(zero(T), false)
end

# Initialize SR1
function init_hessian!(::SR1{T}, cache::RetroCache{T}) where {T}
    return SR1State{T}()
end

# Update SR1 approximation
function update_hessian!(sr1::SR1{T}, state, cache::RetroCache{T}, obj, x) where {T}
    if !state.initialized
        # cache.g already holds the gradient from value_and_gradient!
        copy!(cache.g_prev, cache.g)
        copy!(cache.x_prev, x)
        state.initialized = true
        return
    end
    
    # s = x_k - x_{k-1},  y = g_k - g_{k-1}
    # cache.g already contains g_k (maintained by the optimizer loop)
    @. cache.s = x - cache.x_prev
    @. cache.y = cache.g - cache.g_prev
    
    # Compute H*s using current approximation
    apply_hessian!(cache.tmp, sr1, state, cache, cache.s)
    
    # Compute w = y - H*s
    @. cache.tmp = cache.y - cache.tmp  # w = y - H*s
    
    # Check denominator: w^T * s
    ws = dot(cache.tmp, cache.s)
    
    if abs(ws) > sr1.skip_threshold
        state.α = 1 / ws
    else
        # Skip update
        state.α = zero(T)
    end
    
    # Store for next iteration
    copy!(cache.g_prev, cache.g)
    copy!(cache.x_prev, x)
end

# Apply SR1 Hessian to vector: H * v
function apply_hessian!(Hv, sr1::SR1{T}, state, cache::RetroCache{T}, v) where {T}
    if !state.initialized || abs(state.α) < eps(T)
        # Use scaled identity
        @. Hv = sr1.B0_scale * v
        return
    end
    
    # SR1 formula: H_k = H_{k-1} + (w*w^T)/(w^T*s)
    # where w = y - H_{k-1}*s
    # H*v = H_{k-1}*v + w*(w^T*v)/(w^T*s)
    
    # Start with scaled identity: H_0 * v
    @. Hv = sr1.B0_scale * v
    
    # Compute w = y - H_0*s (stored in cache.tmp from update)
    @. cache.tmp = cache.y - sr1.B0_scale * cache.s
    
    # Add rank-1 update: w*(w^T*v)/(w^T*s)
    wv = dot(cache.tmp, v)
    @. Hv += state.α * wv * cache.tmp
end

# Solve Newton direction: H * d = g, return d
# SR1 is memoryless, so use Sherman-Morrison formula on (H0 + w*w'/ws) 
function solve_newton_direction!(d, sr1::SR1{T}, state, cache::RetroCache{T}, g) where {T}
    if !state.initialized || abs(state.α) < eps(T)
        # H is just scaled identity, so d = g / B0_scale
        @. d = g / sr1.B0_scale
        return true
    end
    
    # Sherman-Morrison: (A + u*v')^{-1} = A^{-1} - A^{-1}*u*v'*A^{-1} / (1 + v'*A^{-1}*u)
    # Here A = B0_scale * I, u = w, v = α * w
    # A^{-1} = (1/B0_scale) * I
    # d = A^{-1}*g - α * (A^{-1}*w) * (w'*A^{-1}*g) / (1 + α * w'*A^{-1}*w)
    
    inv_scale = 1 / sr1.B0_scale
    
    # w = y - B0_scale * s
    @. cache.tmp = cache.y - sr1.B0_scale * cache.s
    
    # A^{-1} * g
    @. d = inv_scale * g
    
    # w' * A^{-1} * g = inv_scale * w' * g
    wTAinvg = inv_scale * dot(cache.tmp, g)
    
    # w' * A^{-1} * w = inv_scale * w' * w
    wTAinvw = inv_scale * dot(cache.tmp, cache.tmp)
    
    # Denominator: 1 + α * w' * A^{-1} * w
    denom = 1 + state.α * wTAinvw
    
    if abs(denom) > eps(T)
        # d = d - α * (A^{-1}*w) * wTAinvg / denom
        factor = state.α * wTAinvg / denom
        @. d -= factor * inv_scale * cache.tmp
        return true
    else
        # Fallback to steepest descent scaled
        @. d = inv_scale * g
        return false
    end
end