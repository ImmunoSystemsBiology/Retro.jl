using Test
using Fides

@testset "Parametric Fallback Types" begin
    # Simple quadratic: f(x) = (x[1] - 1)^2 + (x[2] - 2)^2
    f(x) = (x[1] - 1)^2 + (x[2] - 2)^2
    x0 = [0.0, 0.0]
    
    prob = FidesProblem(f, x0, AutoForwardDiff())
    options = Fides.TrustRegionOptions(verbose=false, maxiter=100)
    
    # Test with CauchyPointFallback (default)
    solver_cauchy = TwoDimSubspace(Fides.CauchyPointFallback())
    result_cauchy = solve(prob, BFGSUpdate(), solver_cauchy; options=options)
    
    @test result_cauchy.converged
    @test result_cauchy.x ≈ [1.0, 2.0] atol=1e-4
    @test result_cauchy.fx < 1e-6
    
    # Test with EigenvalueFallback
    solver_eigen = TwoDimSubspace(Fides.EigenvalueFallback())
    result_eigen = solve(prob, BFGSUpdate(), solver_eigen; options=options)
    
    @test result_eigen.converged
    @test result_eigen.x ≈ [1.0, 2.0] atol=1e-4
    @test result_eigen.fx < 1e-6
    
    # Both should reach similar solutions
    @test result_cauchy.x ≈ result_eigen.x atol=1e-3
end
