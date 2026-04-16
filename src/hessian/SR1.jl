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

mutable struct SR1State{T<:Real}
    α::T  
    initialized::Bool
    
    SR1State{T}() where {T} = new{T}(zero(T), false)
end


function init_hessian!(::SR1{T}, cache::RetroCache{T}) where {T}
    return SR1State{T}()
end


function update_hessian!(sr1::SR1{T}, state::SR1State{T}, cache::RetroCache{T}, ::AbstractObjectiveFunction, x) where {T}
    if !state.initialized
        copy!(cache.g_prev, cache.g)
        copy!(cache.x_prev, x)
        state.initialized = true
        return
    end
    
    @. cache.s = x - cache.x_prev
    @. cache.y = cache.g - cache.g_prev

    apply_hessian!(cache.tmp, sr1, state, cache, cache.s)
    
    @. cache.tmp = cache.y - cache.tmp
    
    ws = dot(cache.tmp, cache.s)
    
    if abs(ws) > sr1.skip_threshold
        state.α = 1 / ws
    else
        state.α = zero(T)
    end

    copy!(cache.g_prev, cache.g)
    copy!(cache.x_prev, x)
end

function apply_hessian!(Hv, sr1::SR1{T}, state::SR1State{T}, cache::RetroCache{T}, v) where {T}
    if !state.initialized || abs(state.α) < eps(T)
        @. Hv = sr1.B0_scale * v
        return
    end
    
    @. Hv = sr1.B0_scale * v

    @. cache.tmp = cache.y - sr1.B0_scale * cache.s

    wv = dot(cache.tmp, v)
    @. Hv += state.α * wv * cache.tmp
end

function solve_newton_direction!(d, sr1::SR1{T}, state::SR1State{T}, cache::RetroCache{T}, g) where {T}
    if !state.initialized || abs(state.α) < eps(T)
        @. d = g / sr1.B0_scale
        return true
    end
    
    inv_scale = 1 / sr1.B0_scale
    @. cache.tmp = cache.y - sr1.B0_scale * cache.s
    @. d = inv_scale * g
    wTAinvg = inv_scale * dot(cache.tmp, g)
    wTAinvw = inv_scale * dot(cache.tmp, cache.tmp)
    denom = 1 + state.α * wTAinvw
    
    if abs(denom) > eps(T)
        factor = state.α * wTAinvg / denom
        @. d -= factor * inv_scale * cache.tmp
        return true
    else
        @. d = inv_scale * g
        return false
    end
end