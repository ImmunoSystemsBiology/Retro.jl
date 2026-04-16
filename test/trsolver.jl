using Retro
using Test
using LinearAlgebra

@testset "Trust-Region Solvers" begin
    @testset "CauchyTRSolver" begin
        solver = CauchyTRSolver()
        n = 3
        p = zeros(n)

        @testset "positive definite H, interior solution" begin
            g = [1.0, 0.0, 0.0]
            H = Matrix{Float64}(I, n, n)
            Delta = 10.0
            α = solve_tr!(solver, g, H, Delta, p)
            # Optimal: α = g'g / g'Hg = 1/1 = 1, within trust region
            @test p ≈ [-1.0, 0.0, 0.0]
            @test α > 0.0
        end

        @testset "positive definite H, boundary solution" begin
            g = [1.0, 0.0, 0.0]
            H = Matrix{Float64}(I, n, n)
            Delta = 0.5
            solve_tr!(solver, g, H, Delta, p)
            # Constrained to boundary
            @test norm(p) ≈ Delta atol=1e-10
        end

        @testset "non-positive-definite H, boundary solution" begin
            g = [1.0, 1.0, 0.0]
            H = -Matrix{Float64}(I, n, n)  # Negative definite
            Delta = 1.0
            solve_tr!(solver, g, H, Delta, p)
            # gHg < 0, go to boundary
            @test norm(p) ≈ Delta atol=1e-10
        end

        @testset "near-zero gradient" begin
            g = fill(1e-20, n)
            H = Matrix{Float64}(I, n, n)
            Delta = 1.0
            α = solve_tr!(solver, g, H, Delta, p)
            # Gradient too small → zero step
            @test all(iszero, p)
            @test α == 0.0
        end
    end

    @testset "EigenTRSolver" begin
        solver = EigenTRSolver()
        n = 3
        p = zeros(n)

        @testset "positive definite H, interior solution" begin
            g = [1.0, 2.0, 3.0]
            H = 10.0 * Matrix{Float64}(I, n, n)  # Large H → small unconstrained step
            Delta = 10.0
            pred = solve_tr!(solver, g, H, Delta, p)
            # Unconstrained step: p = -H\g = -g/10
            @test p ≈ -g ./ 10.0 atol=1e-8
            @test pred > 0.0
        end

        @testset "positive definite H, boundary solution" begin
            g = [1.0, 0.0, 0.0]
            H = Matrix{Float64}(I, n, n)
            Delta = 0.5
            solve_tr!(solver, g, H, Delta, p)
            @test norm(p) ≈ Delta atol=1e-6
        end

        @testset "indefinite H" begin
            g = [1.0, 0.0, 0.0]
            H = Diagonal([-1.0, 2.0, 2.0])
            Delta = 1.0
            pred = solve_tr!(solver, g, H, Delta, p)
            @test norm(p) ≤ Delta + 1e-10
            @test pred >= 0.0
        end

        @testset "near-zero gradient" begin
            g = fill(1e-20, n)
            H = Matrix{Float64}(I, n, n)
            Delta = 1.0
            pred = solve_tr!(solver, g, H, Delta, p)
            @test all(iszero, p)
            @test pred == 0.0
        end
    end
end
