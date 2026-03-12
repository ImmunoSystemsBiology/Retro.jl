```@raw html
<div class="retro-hero">
    <h1>RETRO.JL</h1>
    <div class="tagline">REflective-bounds Trust-Region Optimizer</div>
    <span class="version-badge">v0.0.1 · Julia</span>
</div>

<div class="retro-intro">
    A high-performance, (nearly) allocation-free trust-region optimizer for
    bound-constrained nonlinear problems in Julia.
    Works with <strong>any</strong> AD backend via
    <a href="https://github.com/gdalle/DifferentiationInterface.jl">DifferentiationInterface.jl</a>.
</div>

<div class="retro-iwantto">I want to:</div>

<div class="retro-cards-container">

    <a class="retro-action-card from-left card-optimize"
       href="getting-started/quickstart/">
        <span class="card-number">01</span>
        <div class="card-title">Start optimizing</div>
        <p class="card-desc">
            Install Retro, define a problem, and get a solution in five&nbsp;minutes.
        </p>
        <span class="card-arrow">→ Quick Start</span>
    </a>

    <a class="retro-action-card from-right card-learn"
       href="guide/overview/">
        <span class="card-number">02</span>
        <div class="card-title">Learn about the optimizer</div>
        <p class="card-desc">
            A guided tour from gradient descent to a full trust-region
            reflective algorithm&nbsp;— concept by concept.
        </p>
        <span class="card-arrow">→ Algorithm Tutorial</span>
    </a>

    <a class="retro-action-card from-left card-contribute"
       href="api/problems/">
        <span class="card-number">03</span>
        <div class="card-title">Contribute</div>
        <p class="card-desc">
            Dive into the API reference, architecture notes, and cache
            design to start extending&nbsp;Retro.
        </p>
        <span class="card-arrow">→ API Reference</span>
    </a>

</div>
```

---

## A taste of Retro

```julia
using Retro, ForwardDiff

f(x) = 100*(x[2] - x[1]^2)^2 + (1 - x[1])^2   # Rosenbrock

prob = RetroProblem(f, [-1.2, 1.0], AutoForwardDiff();
                    lb = [-5.0, -5.0], ub = [5.0, 5.0])

result = optimize(prob)

result.x            # ≈ [1.0, 1.0]
is_successful(result)  # true
```

---

## Features at a glance

| Feature | Details |
|:--------|:--------|
| **Bound constraints** | Coleman–Li reflective step with multiple reflections |
| **Hessian strategies** | BFGS (damped), SR1, Exact Hessian via AD |
| **Subspace solvers** | 2-D eigenvalue, Steihaug–Toint CG, full-space |
| **TR solvers** | Eigenvalue, Cauchy, Brent line-search |
| **AD backends** | Any backend via DifferentiationInterface (ForwardDiff, Enzyme, …) |
| **Zero-allocation loops** | Pre-allocated `RetroCache` workspace |
| **Display modes** | `Silent()`, `Iteration()`, `Final()`, `Verbose()` (with ProgressMeter) |

---

## Acknowledgements

Retro.jl is heavily inspired by the [fides](https://github.com/fides-dev/fides)
optimizer in Python, as well as MATLAB's `lsqnonlin` optimizer. 
