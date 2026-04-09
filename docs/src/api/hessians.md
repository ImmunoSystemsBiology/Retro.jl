# API — Hessian Methods

## Abstract type

```@docs
AbstractHessianApproximation
```

## BFGS

```@docs
BFGS
```

## Symmetric Rank-1

```@docs
SR1
```

## Exact Hessian

```@docs
ExactHessian
Retro.ExactHessianState
```

## Common interface

All Hessian approximations implement these methods:

```@docs
init_hessian!
update_hessian!
apply_hessian!
reset_hessian!
```
