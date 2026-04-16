"""
    RetroProblem{OBJ<:AbstractObjectiveFunction, T<:AbstractVector}

Optimization problem definition for Retro.

Encapsulates an objective function, initial point, and optional box constraints.

# Fields
- `objective::OBJ`: Objective function to minimize
- `x0::T`: Initial guess
- `lb::T`: Lower bounds (defaults to `-Inf`)
- `ub::T`: Upper bounds (defaults to `+Inf`)

# Example
```julia
using ForwardDiff
f(x) = sum(abs2, x .- [1.0, 2.0])
prob = RetroProblem(f, [0.0, 0.0], AutoForwardDiff())
```
"""
struct RetroProblem{OBJ<:AbstractObjectiveFunction, T<:AbstractVector}
    objective::OBJ
    x0::T
    lb::T
    ub::T

    function RetroProblem(obj::OBJ, x0::T, lb::T, ub::T) where {OBJ<:AbstractObjectiveFunction, T<:AbstractVector}
        isempty(x0) && throw(ArgumentError("Initial guess x0 cannot be empty"))
        length(lb) != length(x0) && throw(ArgumentError("Length of lower bounds must match length of x0"))
        length(ub) != length(x0) && throw(ArgumentError("Length of upper bounds must match length of x0"))
        any(lb .>= ub) && throw(ArgumentError("Each lower bound must be strictly less than corresponding upper bound"))
        new{OBJ, T}(obj, x0, lb, ub)
    end
end

function RetroProblem(func::Function, x0::AbstractVector{T}, adtype::AbstractADType;
                     lb::AbstractVector = fill(eltype(x0)(-Inf), length(x0)),
                     ub::AbstractVector = fill(eltype(x0)(Inf), length(x0))) where {T<:Real}
    isempty(x0) && throw(ArgumentError("Initial guess x0 cannot be empty"))
    obj = ADObjectiveFunction(func, adtype, x0)
    RetroProblem(obj, x0, T.(lb), T.(ub))
end

function RetroProblem(func::Function, grad!::Function, x0::AbstractVector{T}, adtype::AbstractADType;
                     lb::AbstractVector = fill(eltype(x0)(-Inf), length(x0)),
                     ub::AbstractVector = fill(eltype(x0)(Inf), length(x0))) where {T<:Real}
    obj = GradientObjectiveFunction(func, grad!, adtype, x0)
    RetroProblem(obj, x0, T.(lb), T.(ub))
end

function RetroProblem(func::Function, grad!::Function, hess!::Function, x0::AbstractVector{T};
                     lb::AbstractVector = fill(eltype(x0)(-Inf), length(x0)),
                     ub::AbstractVector = fill(eltype(x0)(Inf), length(x0))) where {T<:Real}
    obj = AnalyticObjectiveFunction(func, grad!, hess!)
    RetroProblem(obj, x0, T.(lb), T.(ub))
end