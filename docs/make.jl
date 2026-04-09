using Documenter
using Retro

const is_live = "--live" in ARGS

makedocs(
    sitename = "Retro.jl",
    modules  = [Retro],
    format   = Documenter.HTML(
        prettyurls = !is_live && get(ENV, "CI", nothing) == "true",
        canonical  = "https://immunosystemsbiology.github.io/Retro.jl/stable",
        assets     = ["assets/custom.css", "assets/custom.js"],
        collapselevel = 1,
    ),
    pages = [
        "Home" => "index.md",

        "Getting Started" => [
            "Quick Start"  => "getting-started/quickstart.md",
            "Examples"     => "getting-started/examples.md",
        ],

        "Algorithm Tutorial" => [
            "Overview"                => "guide/overview.md",
            "The Simplest Optimizer"  => "guide/basic-optimizer.md",
            "Adding Trust Regions"    => "guide/trust-region.md",
            "Handling Bounds"         => "guide/reflective-bounds.md",
            "Working in Subspaces"    => "guide/subspaces.md",
            "Hessian Approximations"  => "guide/hessian-approximations.md",
            "Robustness & Fallbacks"  => "guide/robustness.md",
        ],

        "How-To Guides" => [
            "Bounded Problems"           => "howto/bounded-problems.md",
            "Automatic Differentiation"  => "howto/autodiff.md",
            "Large-Scale Problems"       => "howto/large-scale.md",
        ],

        "API Reference" => [
            "Problems & Results"  => "api/problems.md",
            "Objective Functions" => "api/objectives.md",
            "Hessian Methods"    => "api/hessians.md",
            "Subspace Methods"   => "api/subspaces.md",
            "TR Solvers"         => "api/solvers.md",
            "Bounds & Reflection" => "api/bounds.md",
            "Options & Display"  => "api/options.md",
        ],

        "Internals" => [
            "Architecture"            => "internals/architecture.md",
            "Trust-Region Dynamics"   => "internals/trust-region-dynamics.md",
            "Cache & Allocation Design" => "internals/cache-design.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/immunosystemsbiology/Retro.jl.git",
    devbranch = "main",
)

if is_live
    using LiveServer
    serve(dir = joinpath(@__DIR__, "build"))
end
