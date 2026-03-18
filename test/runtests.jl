using Retro
using Test
using LinearAlgebra
using Random

Random.seed!(1234)

@testset "Retro.jl Tests" begin
    @testset "Boilerplate Tests" begin
        include("boilerplate.jl")
    end

    @testset "Error Call Tests" begin
        include("error_calls.jl")
    end

    @testset "Differentiation Interface Tests" begin
        include("differentiation.jl")
    end

    @testset "Rosenbrock Problem" begin
        include("rosenbrock.jl")
    end

    @testset "Challenging Optimization Problems" begin
        include("challenging_problems.jl")
    end

    @testset "ODE Parameter Estimation Problems" begin
        include("ode.jl")
    end
end

