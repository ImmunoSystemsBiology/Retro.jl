```
██████╗ ███████╗████████╗██████╗  ██████╗ 
██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗
██████╔╝█████╗     ██║   ██████╔╝██║   ██║
██╔══██╗██╔══╝     ██║   ██╔══██╗██║   ██║
██║  ██║███████╗   ██║   ██║  ██║╚██████╔╝
╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝
Reflective-bounds Trust Region Optimizer
```

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://max-de-rooij.com/Retro.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://max-de-rooij.com/Retro.jl/dev)
![License](https://img.shields.io/badge/License-BSD--3-green?labelColor=white&link=https%3A%2F%2Fgithub.com%2Fmax-de-rooij%2FRetro.jl%2Fblob%2Fmain%2FLICENSE)
[![tests](https://github.com/max-de-rooij/Retro.jl/actions/workflows/tests.yml/badge.svg)](https://github.com/max-de-rooij/Retro.jl/actions/workflows/tests.yml)

Retro (REflective-bounds Trust-Region Optimizer): A high-performance Julia package for bound-constrained optimization using trust-region reflective methods.

> [!WARNING]
> The optimizer is still under development, and has not been fully tested, yet. Use with caution.

## Features

- **Multiple Hessian Approximations**: BFGS, SR1, Exact Hessian, and Gauss-Newton
- **Flexible Subproblem Solvers**: 2D subspace, Conjugate Gradient, and full-space eigenvalue solvers
- **Bound Constraints**: Interior-point reflective method (Coleman-Li algorithm)
- **Automatic Differentiation**: Seamless integration via DifferentiationInterface
- **Least-Squares Support**: Specialized Gauss-Newton for residual formulations
  
## Quick Start

### Adding Retro to your project
1. Make a local copy of Retro on your machine and store it in a location that is easily accessible.
2. In your Julia project, run `pkg> develop <path/to/Retro.jl>`
3. Retro will be usable within your project.

### Using Retro

```julia
using Retro, ForwardDiff

# Define a scalar objective f : ℝⁿ → ℝ
f(x) = (x[1] - 3.0)^2 + (x[2] + 1.0)^2

# Wrap it in a problem: function, initial guess, AD backend
prob = RetroProblem(f, [0.0, 0.0], AutoForwardDiff())

# Optimize
result = optimize(prob)
```

## Hessian Strategies

- `BFGSUpdate()`: Quasi-Newton BFGS (default; recommended for general use)
- `SR1Update()`: Symmetric Rank-1 (good for indefinite problems)
- `ExactHessian()`: Exact Hessian via AD (expensive but accurate)

## Subproblem Solvers

- `TwoDimSubspace()`: 2D subspace method (default; recommended, good balance)
- `CGSubspace()`: Steihaug-Toint CG (good for large problems)
- `FullSpace()`: Eigenvalue decomposition (most accurate, expensive)

## Acknowledgements
Retro.jl is heavily inspired by the [fides](https://github.com/fides-dev/fides) optimizer in Python, as well as the MATLAB `lsqnonlin` optimizer.
