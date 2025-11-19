using Fides, Test, BenchmarkTools

@testset "Boilerplate Tests" begin
    include("boilerplate.jl")
end

@testset "Differentiation Interface Tests" begin
    include("differentiation.jl")
end

@testset "Rosenbrock Problem" begin
    include("rosenbrock.jl")
end

@testset "Parametric Fallback Types" begin
    include("fallback_test.jl")
end