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

        @testset "Utility Functions" begin
            include("utils.jl")
        end

        @testset "TR Solver Unit Tests" begin
            include("trsolver.jl")
        end

        @testset "Display Modes and RetroResult" begin
            include("displays.jl")
        end

        @testset "FullSpace and Solver Integration" begin
            include("solvers.jl")
        end
end

