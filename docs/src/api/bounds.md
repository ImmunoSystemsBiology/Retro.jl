# API — Bounds & Reflection

Internal helpers for Coleman–Li reflective bound handling.  These are
called automatically during [`optimize`](@ref); you only need them if you
are extending Retro or writing custom step logic.

## Scaling

```@docs
compute_scaling!
compute_affine_scaling!
scale_gradient!
```

## Feasibility

```@docs
initialize_away_from_bounds!
project_bounds!
find_step_to_bound
```

## Reflection

```@docs
Retro.reflect_step!
apply_reflective_bounds!
compute_cauchy_boundary_point!
projected_gradient_norm
```
