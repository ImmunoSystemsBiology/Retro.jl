using Retro
using Test
using LinearAlgebra

# Simple quadratic: f(x) = ||x - [1, 2]||^2, minimum at [1, 2]
_quad_f(x) = sum((x .- [1.0, 2.0]).^2)

@testset "FullSpace Subspace" begin
    prob = RetroProblem(_quad_f, [0.0, 0.0], AutoForwardDiff())

    @testset "FullSpace + EigenTRSolver converges" begin
        result = optimize(prob; subspace=FullSpace(), tr_solver=EigenTRSolver(), display=Silent())
        @test is_successful(result)
        @test result.x ≈ [1.0, 2.0] atol=1e-4
    end

    @testset "FullSpace + CauchyTRSolver converges" begin
        result = optimize(prob; subspace=FullSpace(), tr_solver=CauchyTRSolver(), display=Silent())
        @test is_successful(result)
        @test result.x ≈ [1.0, 2.0] atol=1e-4
    end

    @testset "FullSpace with ExactHessian" begin
        result = optimize(prob;
            subspace=FullSpace(),
            tr_solver=EigenTRSolver(),
            hessian_approximation=ExactHessian(),
            display=Silent()
        )
        @test is_successful(result)
        @test result.x ≈ [1.0, 2.0] atol=1e-4
    end
end

@testset "CauchyTRSolver via optimize" begin
    prob = RetroProblem(_quad_f, [0.0, 0.0], AutoForwardDiff())
    result = optimize(prob; subspace=FullSpace(), tr_solver=CauchyTRSolver(), display=Silent())
    @test result isa RetroResult
    @test is_successful(result)
end
