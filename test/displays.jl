using Retro
using Test
using LinearAlgebra
@testset "RetroResult" begin
    x = [1.0, 2.0]
    gx = [0.0, 0.0]

    @testset "is_successful with converged reasons" begin
        for reason in (:gtol, :ftol, :xtol)
            r = RetroResult(x, 0.5, gx, 10, 15, 12, 5, reason)
            @test is_successful(r) == true
        end
    end

    @testset "is_successful with non-converged reasons" begin
        for reason in (:maxiter, :stagnation, :tr_radius_too_small)
            r = RetroResult(x, 0.5, gx, 10, 15, 12, 5, reason)
            @test is_successful(r) == false
        end
    end

    @testset "Base.show" begin
        r = RetroResult(x, 1.23, gx, 5, 10, 8, 3, :gtol)
        buf = IOBuffer()
        show(buf, r)
        s = String(take!(buf))
        @test occursin("RetroResult", s)
        @test occursin("1.23", s)
        @test occursin("gtol", s)
        @test occursin("5", s)   # iterations
    end
end

@testset "Display Modes" begin
    f(x) = sum(abs2, x .- 1.0)
    x0 = [0.0, 0.0]
    prob = RetroProblem(f, x0, AutoForwardDiff())

    @testset "display modes run without error" begin
        for mode in (Silent(), Final(), Iteration(), Verbose(), Debug())
            result = optimize(prob; maxiter=10, display=mode)
            @test result isa RetroResult
        end
    end
end
