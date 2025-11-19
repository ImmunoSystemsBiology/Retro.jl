import ReverseDiff, Zygote, FiniteDifferences

# define simple quadratic function
square(x) = sum(x.^2)
x0 = [1.0, 1.0]

@testset "AutoForwardDiff" begin
    prob = FidesProblem(square, x0, AutoForwardDiff())
    result = Fides.solve(prob, Fides.BFGSUpdate(), Fides.TwoDimSubspace())
    @test result.converged
    @test isapprox(result.x, [0.0, 0.0]; atol=1e-3)
    @test isapprox(result.fx, 0.0; atol=1e-6)
end

@testset "AutoReverseDiff" begin
    prob = FidesProblem(square, x0, AutoReverseDiff())
    result = Fides.solve(prob, Fides.BFGSUpdate(), Fides.TwoDimSubspace())
    @test result.converged
    @test isapprox(result.x, [0.0, 0.0]; atol=1e-3)
    @test isapprox(result.fx, 0.0; atol=1e-6)
end

@testset "AutoFiniteDifferences" begin
    prob = FidesProblem(square, x0, AutoFiniteDifferences())
    result = Fides.solve(prob, Fides.BFGSUpdate(), Fides.TwoDimSubspace())
    @test result.converged
    @test isapprox(result.x, [0.0, 0.0]; atol=1e-3)
    @test isapprox(result.fx, 0.0; atol=1e-6)
end

@testset "AutoZygote" begin
    prob = FidesProblem(square, x0, AutoZygote())
    result = Fides.solve(prob, Fides.BFGSUpdate(), Fides.TwoDimSubspace())
    @test result.converged
    @test isapprox(result.x, [0.0, 0.0]; atol=1e-3)
    @test isapprox(result.fx, 0.0; atol=1e-6)
end


