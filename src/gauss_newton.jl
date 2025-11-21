"""
    Gauss-Newton Hessian Approximations

Special Hessian approximations for nonlinear least-squares problems,
which commonly arise in ODE parameter estimation.

For a least-squares objective f(x) = 0.5 * ||r(x)||², the Hessian can be
approximated as H ≈ J'J where J is the Jacobian of the residuals. This is
the Gauss-Newton approximation and works well when residuals are small.
"""



"""
    compute_gauss_newton_hessian(resfun, adtype, x, params...)

Compute Gauss-Newton Hessian approximation H = J'*J where J is the residual Jacobian.

For a least-squares objective f(x) = 0.5*||r(x)||², the Hessian can be approximated
as H ≈ J'*J, which ignores second-order terms but is accurate when residuals are small.

Additional arguments (e.g., Constant(p)) are passed through to the residual function.
"""
function compute_gauss_newton_hessian(resfun, adtype, x, params...)
    # Compute Jacobian of residuals
    prep_jac = prepare_jacobian(resfun, adtype, x, params...)
    r, jac = value_and_jacobian(resfun, prep_jac, adtype, x, params...)
    
    # Form Gauss-Newton approximation: H = J'*J
    H = jac' * jac
    
    # Compute gradient: g = J'*r (for f = 0.5*||r||²)
    g = jac' * r
    
    # Compute function value: f = 0.5*||r||²
    f = 0.5 * dot(r, r)
    
    return f, g, H
end

"""
    HybridUpdate{Initial<:AbstractHessianUpdate, Fallback<:AbstractHessianUpdate} <: AbstractHessianUpdate

Hybrid Hessian update strategy that switches from initial to fallback method based on trust-region radius updates.

Following the Fides paper, this strategy starts with an initial Hessian approximation (typically Gauss-Newton)
and simultaneously constructs a fallback approximation (typically BFGS). The strategy switches to the fallback
when the quality of the initial approximation becomes limiting, as determined by failure to update the 
trust-region radius for `nhybrid` consecutive iterations.

This design leverages the high success rate of Gauss-Newton in early optimization (determining basin of 
attraction) while benefiting from BFGS's higher convergence rate in later phases (accurate Hessian approximation).

# Fields
- `initial::Initial`: Initial Hessian update strategy (typically GaussNewtonUpdate)
- `fallback::Fallback`: Fallback strategy for later optimization (typically BFGSUpdate)
- `nhybrid::Int`: Number of consecutive iterations without radius update before switching (default: 5)
- `switched::Ref{Bool}`: Whether we've switched to fallback (internal state)
- `no_radius_update_count::Ref{Int}`: Counter for iterations without radius update (internal state)

# Example
```julia
# Gauss-Newton to BFGS hybrid - switches after 5 iterations without radius update
residuals(x, p) = x .- p.target
objective(x, p) = 0.5 * sum(abs2, residuals(x, p))
prob = RetroProblem(objective, x0, AutoForwardDiff())
hybrid = HybridUpdate(GaussNewtonUpdate(residuals), BFGSUpdate(), 5)
result = solve(prob, hybrid, TwoDimSubspace(), p)

# Exact Hessian to BFGS hybrid
hybrid = HybridUpdate(ExactHessian(), BFGSUpdate(), 3)
result = solve(prob, hybrid, TwoDimSubspace())
```
"""
mutable struct HybridUpdate{Initial<:AbstractHessianUpdate, Fallback<:AbstractHessianUpdate} <: AbstractHessianUpdate
    initial::Initial
    fallback::Fallback
    nhybrid::Int
    switched::Ref{Bool}
    no_radius_update_count::Ref{Int}
    
    function HybridUpdate(initial::Initial, fallback::Fallback, nhybrid::Int=5) where {Initial, Fallback}
        new{Initial, Fallback}(initial, fallback, nhybrid, Ref(false), Ref(0))
    end
end

# Helper function to reset hybrid counter when radius is updated
function reset_hybrid_on_radius_update!(hybrid::HybridUpdate)
    hybrid.no_radius_update_count[] = 0
    nothing
end

# Helper function to record iteration without radius update and check if we should switch
function record_no_radius_update!(hybrid::HybridUpdate)
    hybrid.no_radius_update_count[] += 1
    if hybrid.no_radius_update_count[] >= hybrid.nhybrid
        hybrid.switched[] = true
    end
    nothing
end

# Get current active strategy
function current_strategy(hybrid::HybridUpdate)
    return hybrid.switched[] ? hybrid.fallback : hybrid.initial
end

