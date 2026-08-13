# Internals — Architecture

This page is for **contributors and researchers** who want to understand how
Retro's source code is organized.

## Module structure

```
src/
├── Retro.jl                 # Module root: imports, includes, exports
├── types.jl                 # Abstract type hierarchy
├── objective.jl             # ADObjectiveFunction, gradient!, hessian!, …
├── optimize.jl              # Main loop, RetroOptions, convergence checks
│
├── problem/
│   ├── RetroProblem.jl      # Problem struct + convenience constructors
│   ├── RetroCache.jl        # Pre-allocated workspace
│   └── RetroResult.jl       # Immutable result container
│
├── hessian/
│   ├── BFGS.jl              # Damped BFGS with Powell damping
│   ├── SR1.jl               # Memoryless symmetric rank-1
│   └── ExactHessian.jl      # Full Hessian via DI with caching
│
├── subspace/
│   ├── TwoDimSubspace.jl    # 2-D eigenvalue TR solve (StaticArrays)
│   ├── CGSubspace.jl        # Steihaug–Toint truncated CG
│   └── FullSpace.jl         # Full n-D eigendecomposition
│
├── trsolver/
│   ├── EigenTRSolver.jl     # Eigenvalue TR solver (Newton → Brent → Cauchy)
│   └── CauchyTRSolver.jl    # Cauchy (steepest descent) point
│
├── steps/
│   ├── Reflection.jl        # Coleman–Li reflective bounds
│   ├── StepAcceptance.jl    # ρ ratio, radius update, convergence
│   └── TrustRegionStep.jl   # Orchestrates subspace → TR solve → reflection
│
└── utils/
    ├── LinearAlgebraHelpers.jl   # safe_norm, safe_dot, etc.
    ├── Norms.jl                  # Specialised norm functions
    └── Displays.jl               # Silent/Iteration/Final/Verbose + ProgressMeter
```

## Type hierarchy

```
AbstractObjectiveFunction
├── ADObjectiveFunction          (AD gradient + Hessian)
├── GradientObjectiveFunction    (user gradient, AD Hessian)
└── AnalyticObjectiveFunction    (all user-supplied)

AbstractHessianApproximation
├── BFGS
├── SR1
└── ExactHessian

AbstractSubspace
├── TwoDimSubspace
├── CGSubspace
└── FullSpace

AbstractTRSolver
├── EigenTRSolver
└── CauchyTRSolver

AbstractDisplayMode
├── Silent
├── Iteration
├── Final
└── Verbose
```

## One iteration, step by step

The `optimize` loop in [optimize.jl](https://github.com/max-de-rooij/Retro.jl/blob/main/src/optimize.jl)
does the following on every iteration:

```
1.  check_convergence(g, p, …, options)
        → return if converged

2.  update_hessian!(hess_approx, state, cache, obj, x)
        → BFGS rank-2 update  /  SR1 rank-1 update  /  exact Hessian eval

3.  compute_trust_region_step!(cache, prob, subspace, …, x, Δ, options)
        a. compute_scaling!(scaling, x, lb, ub, θ₁, θ₂)
        b. scale_gradient!(scaled_g, g, scaling)
        c. build_subspace!(subspace, state, cache, hess, …, x)
        d. solve_subspace_tr!(solver, subspace, state, cache, Δ)
        e. apply inverse scaling to step p
        f. apply_reflective_bounds!(x_trial, x, p, lb, ub, θ₂; g)

4.  compute_hv_product!(tmp, hess, state, cache, p)
5.  predicted_reduction(g, p, tmp)
6.  f_trial = objfunc!(cache, obj, x_trial)
7.  actual_reduction(f_current, f_trial, p, g_trial, dv)
8.  ρ = actual / predicted

9.  if accept_step(ρ, μ):
        x ← x_trial
        f_current ← value_and_gradient!(g, cache, obj, x)

10. Δ ← update_trust_region_radius(Δ, ρ, step_norm, μ, η, γ₁, γ₂, Δ_max)
```

## Dispatch pattern

Retro uses **Julia's multiple dispatch** extensively:

* The *Hessian approximation* type controls how `update_hessian!`,
  `apply_hessian!`, and `solve_newton_direction!` behave.
* The *subspace* type controls `build_subspace!` and `solve_subspace_tr!`.
* The *TR solver* type controls the inner eigenvalue / Cauchy / solver
  via `solve_tr!` and `solve_tr_2d!`.
* The *objective* type controls whether gradient / Hessian calls hit AD or
  user functions.

This means **adding a new Hessian method or subspace is a matter of defining
a new struct and implementing the interface methods** — no modification to the
main loop required.

## Key design decisions

| Decision | Rationale |
|:---------|:----------|
| Single `RetroCache` struct | All working vectors pre-allocated once; passed down the call tree |
| `StaticArrays` for 2-D subspace | Zero allocation for the inner TR solve |
| `DifferentiationInterface` for AD | Backend-agnostic; prep objects cached at construction |
| Separate `prep_g` and `prep_h` | Gradient and Hessian may use different DI code paths |
| Reflections in a loop (not recursive) | Bounded stack depth; easy to set a max-reflection limit |
