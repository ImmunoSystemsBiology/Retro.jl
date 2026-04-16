"""
    ADObjectiveFunction{F,ADT,PG,PH} <: AbstractObjectiveFunction

Objective function with automatic differentiation via DifferentiationInterface.
Precomputes gradient and Hessian preparation objects at construction time so that
repeated evaluations are allocation-free.

Works with any AD backend that DifferentiationInterface supports 
(ForwardDiff, Enzyme, Zygote, …) as long as the objective is compatible.

# Fields
- `func::F`: Scalar objective ``f : \\mathbb{R}^n \\to \\mathbb{R}``
- `adtype::ADT`: AD backend (e.g. `AutoForwardDiff()`)
- `prep_g::PG`: Prepared gradient operator (from `prepare_gradient`)
- `prep_h::PH`: Prepared Hessian operator  (from `prepare_hessian`)
"""
struct ADObjectiveFunction{F,ADT,PG,PH} <: AbstractObjectiveFunction
    func::F
    adtype::ADT
    prep_g::PG
    prep_h::PH

    function ADObjectiveFunction(func::F, adtype::ADT, x0) where {F,ADT}
        prep_g = prepare_gradient(func, adtype, x0)
        prep_h = prepare_hessian(func, adtype, x0)
        new{F,ADT,typeof(prep_g),typeof(prep_h)}(func, adtype, prep_g, prep_h)
    end
end

"""
    GradientObjectiveFunction{F,G,ADT,PH} <: AbstractObjectiveFunction

Objective with a user-supplied gradient and AD-computed Hessian.

# Fields
- `func::F`: Scalar objective ``f : \\mathbb{R}^n \\to \\mathbb{R}``
- `grad!::G`: In-place gradient `grad!(g, x)`
- `adtype::ADT`: AD backend used only for Hessian computation
- `prep_h::PH`: Prepared Hessian operator
"""
struct GradientObjectiveFunction{F,G,ADT,PH} <: AbstractObjectiveFunction
    func::F
    grad!::G
    adtype::ADT
    prep_h::PH

    function GradientObjectiveFunction(func::F, grad!::G, adtype::ADT, x0) where {F,G,ADT}
        prep_h = prepare_hessian(func, adtype, x0)
        new{F,G,ADT,typeof(prep_h)}(func, grad!, adtype, prep_h)
    end
end

"""
    AnalyticObjectiveFunction{F,G,H} <: AbstractObjectiveFunction

Fully user-supplied objective, gradient, and Hessian. No AD dependency.

# Fields
- `func::F`: Scalar objective ``f : \\mathbb{R}^n \\to \\mathbb{R}``
- `grad!::G`: In-place gradient `grad!(g, x)`
- `hess!::H`: In-place Hessian  `hess!(H, x)`
"""
struct AnalyticObjectiveFunction{F,G,H} <: AbstractObjectiveFunction
    func::F
    grad!::G
    hess!::H
end

"""
    objfunc!(cache, obj, x) -> f(x)

Evaluate the objective function at `x`, incrementing the call counter in `cache`.
"""
function objfunc!(cache::RetroCache, obj::ADObjectiveFunction, x)
    cache.f_calls += 1
    return obj.func(x)
end

function objfunc!(cache::RetroCache, obj::GradientObjectiveFunction, x)
    cache.f_calls += 1
    return obj.func(x)
end

function objfunc!(cache::RetroCache, obj::AnalyticObjectiveFunction, x)
    cache.f_calls += 1
    return obj.func(x)
end

"""
    gradient!(g, cache, obj, x)

Compute the gradient of `obj` at `x` and store it in `g` (in-place).
"""
function DifferentiationInterface.gradient!(g, cache::RetroCache, obj::ADObjectiveFunction, x)
    cache.g_calls += 1
    DifferentiationInterface.gradient!(obj.func, g, obj.prep_g, obj.adtype, x)
end

function DifferentiationInterface.gradient!(g, cache::RetroCache, obj::GradientObjectiveFunction, x)
    cache.g_calls += 1
    obj.grad!(g, x)
end

function DifferentiationInterface.gradient!(g, cache::RetroCache, obj::AnalyticObjectiveFunction, x)
    cache.g_calls += 1
    obj.grad!(g, x)
end

"""
    value_and_gradient!(g, cache, obj, x) -> f(x)

Compute objective value and gradient simultaneously.  Returns the scalar
objective value; the gradient is written to `g`.
"""
function DifferentiationInterface.value_and_gradient!(g, cache::RetroCache, obj::ADObjectiveFunction, x)
    cache.f_calls += 1
    cache.g_calls += 1
    val, _ = DifferentiationInterface.value_and_gradient!(obj.func, g, obj.prep_g, obj.adtype, x)
    return val
end

function DifferentiationInterface.value_and_gradient!(g, cache::RetroCache, obj::GradientObjectiveFunction, x)
    cache.f_calls += 1
    cache.g_calls += 1
    val = obj.func(x)
    obj.grad!(g, x)
    return val
end

function DifferentiationInterface.value_and_gradient!(g, cache::RetroCache, obj::AnalyticObjectiveFunction, x)
    cache.f_calls += 1
    cache.g_calls += 1
    val = obj.func(x)
    obj.grad!(g, x)
    return val
end

"""
    hessian!(H, cache, obj, x)

Compute the Hessian of `obj` at `x` and store it in the matrix `H` (in-place).
"""
function DifferentiationInterface.hessian!(H, cache::RetroCache, obj::ADObjectiveFunction, x)
    cache.h_calls += 1
    DifferentiationInterface.hessian!(obj.func, H, obj.prep_h, obj.adtype, x)
end

function DifferentiationInterface.hessian!(H, cache::RetroCache, obj::GradientObjectiveFunction, x)
    cache.h_calls += 1
    DifferentiationInterface.hessian!(obj.func, H, obj.prep_h, obj.adtype, x)
end

function DifferentiationInterface.hessian!(H, cache::RetroCache, obj::AnalyticObjectiveFunction, x)
    cache.h_calls += 1
    obj.hess!(H, x)
end

"""
    value_gradient_and_hessian!(g, H, cache, obj, x) -> f(x)

Compute objective value, gradient, and Hessian simultaneously.
Returns the scalar objective value; `g` and `H` are written in-place.
"""
function DifferentiationInterface.value_gradient_and_hessian!(g, H, cache::RetroCache, obj::ADObjectiveFunction, x)
    cache.f_calls += 1
    cache.g_calls += 1
    cache.h_calls += 1
    val, _ = DifferentiationInterface.value_gradient_and_hessian!(
        obj.func, g, H, obj.prep_h, obj.adtype, x
    )
    return val
end

function DifferentiationInterface.value_gradient_and_hessian!(g, H, cache::RetroCache, obj::GradientObjectiveFunction, x)
    cache.f_calls += 1
    cache.g_calls += 1
    cache.h_calls += 1
    val = obj.func(x)
    obj.grad!(g, x)
    DifferentiationInterface.hessian!(obj.func, H, obj.prep_h, obj.adtype, x)
    return val
end

function DifferentiationInterface.value_gradient_and_hessian!(g, H, cache::RetroCache, obj::AnalyticObjectiveFunction, x)
    cache.f_calls += 1
    cache.g_calls += 1
    cache.h_calls += 1
    val = obj.func(x)
    obj.grad!(g, x)
    obj.hess!(H, x)
    return val
end