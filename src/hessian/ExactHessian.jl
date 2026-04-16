"""
    ExactHessian{T} <: AbstractHessianApproximation

Exact Hessian computed via automatic differentiation (or user-supplied analytic Hessian).
Adds a small diagonal regularization to improve conditioning.

Best for small-to-medium problems where the full Hessian is affordable.

# Fields
- `regularization::T`: Added to diagonal of H for numerical stability (default: 1e-8)
"""
struct ExactHessian{T<:Real} <: AbstractHessianApproximation
    regularization::T
    
    ExactHessian{T}(; regularization::T = T(1e-8)) where {T} = new{T}(regularization)
end

ExactHessian(; kwargs...) = ExactHessian{Float64}(; kwargs...)

"""
    ExactHessianState{T}

Cached Hessian matrix and the point where it was last computed.
Recomputes only when `x` changes.
"""
mutable struct ExactHessianState{T<:Real, M<:AbstractMatrix{T}}
    H::M
    x_cached::Vector{T}
    valid::Bool
    
    ExactHessianState{T}(n::Int) where {T} = new{T, Matrix{T}}(zeros(T, n, n), zeros(T, n), false)
end

function init_hessian!(::ExactHessian{T}, cache::RetroCache{T}) where {T}
    n = length(cache.g)
    return ExactHessianState{T}(n)
end

function update_hessian!(eh::ExactHessian{T}, state::ExactHessianState{T}, cache::RetroCache{T}, obj::AbstractObjectiveFunction, x) where {T}
    if !state.valid || norm(x - state.x_cached) > eps(T)
        try
            hessian!(state.H, cache, obj, x)
            
            for i in 1:size(state.H, 1)
                state.H[i, i] += eh.regularization
            end
            
            copy!(state.x_cached, x)
            state.valid = true
        catch 
            fill!(state.H, zero(T))
            for i in 1:size(state.H, 1)
                state.H[i, i] = one(T)
            end
            state.valid = false
        end
    end
end

function apply_hessian!(Hv, ::ExactHessian{T}, state::ExactHessianState{T}, cache::RetroCache{T}, v) where {T}
    if state.valid
        mul!(Hv, state.H, v)
    else
        copy!(Hv, v)
    end
end

function solve_newton_direction!(d, ::ExactHessian{T}, state::ExactHessianState{T}, cache::RetroCache{T}, g) where {T}
    if !state.valid
        copy!(d, g)
        return false
    end
    
    try
        F = cholesky(Symmetric(state.H), check=false)
        if issuccess(F)
            d .= F \ g
            return true
        end
    catch
    end
    
    try
        d .= state.H \ g
        return true
    catch
        copy!(d, g)
        return false
    end
end