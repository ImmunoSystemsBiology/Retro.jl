"""
    CustomHessian{T} <: AbstractHessianApproximation

Supply your own Hessian computation function.

# Fields
- `hfunc!`: Function to compute the Hessian in place (signature hfun!(H, x))
"""
struct CustomHessian <: AbstractHessianApproximation
    hfunc!::Function
end

"""
    CustomHessianState{T}

Cached Hessian matrix and the point where it was last computed.
Recomputes only when `x` changes.
"""
mutable struct CustomHessianState{T<:Real, M<:AbstractMatrix{T}}
    H::M
    x_cached::Vector{T}
    valid::Bool
    
    CustomHessianState{T}(n::Int) where {T} = new{T, Matrix{T}}(zeros(T, n, n), zeros(T, n), false)
end

function init_hessian!(::CustomHessian, cache::RetroCache{T}) where {T}
    n = length(cache.g)
    return CustomHessianState{T}(n)
end

function update_hessian!(eh::CustomHessian, state::CustomHessianState{T}, ::RetroCache{T}, ::AbstractObjectiveFunction, x) where {T}
    if !state.valid || norm(x - state.x_cached) > eps(T)
        try
            eh.hfunc!(state.H, x)
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

function apply_hessian!(Hv, ::CustomHessian, state::CustomHessianState{T}, ::RetroCache{T}, v) where {T}
    if state.valid
        mul!(Hv, state.H, v)
    else
        copy!(Hv, v)
    end
end

function solve_newton_direction!(d, ::CustomHessian, state::CustomHessianState{T}, ::RetroCache{T}, g) where {T}
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