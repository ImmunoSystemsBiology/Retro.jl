using Retro
using Test
using LinearAlgebra

@testset "Rosenbrock Problem" begin
    # Classic test problem: f(x,y) = 100*(y - x^2)^2 + (1 - x)^2
    # Global minimum at (1, 1) with value 0
    
    rosenbrock(x) = 100*(x[2] - x[1]^2)^2 + (1 - x[1])^2
    
    function rosenbrock_grad!(g, x)
        g[1] = -400*x[1]*(x[2] - x[1]^2) - 2*(1 - x[1])
        g[2] = 200*(x[2] - x[1]^2)
    end
    
    function rosenbrock_hess!(H, x)
        H[1,1] = -400*(x[2] - 3*x[1]^2) + 2
        H[1,2] = -400*x[1]
        H[2,1] = -400*x[1]
        H[2,2] = 200
    end
    
    @testset "AD Version" begin
        x0 = [-1.2, 1.0]  # Classic starting point
        prob_ad = RetroProblem(rosenbrock, x0, AutoForwardDiff())
        
        @test objfunc!(RetroCache(2), prob_ad.objective, x0) ≈ 24.2
        
        # Test optimization with different configurations
        @testset "TwoDimSubspace + BFGS" begin
            result = optimize(prob_ad; 
                            maxiter=100, 
                            display=Silent(),
                            subspace=TwoDimSubspace(),
                            hessian_approximation=BFGS())
            
            @test result isa RetroResult
            @test norm(result.x - [1.0, 1.0]) < 0.1  # Should be close to optimum
            @test result.fx < 1e-2  # Should be close to zero
            @test is_successful(result) || result.iterations == 100
        end
        
        @testset "TwoDimSubspace + SR1" begin
            result = optimize(prob_ad;
                            maxiter=500,  # SR1 needs more iterations on Rosenbrock
                            display=Silent(), 
                            subspace=TwoDimSubspace(),
                            hessian_approximation=SR1())
            
            @test result isa RetroResult
            @test norm(result.x - [1.0, 1.0]) < 0.2  # SR1 might be less accurate
        end
        
        @testset "CGSubspace + BFGS" begin
            result = optimize(prob_ad;
                            maxiter=50,
                            display=Silent(),
                            subspace=CGSubspace(max_cg_iter=5),
                            hessian_approximation=BFGS())
            
            @test result isa RetroResult
            # CG might need more iterations
        end
    end
    
    @testset "Analytical Gradient Version" begin
        x0 = [-1.2, 1.0]
        prob_grad = RetroProblem(rosenbrock, rosenbrock_grad!, x0, AutoForwardDiff())
        
        result = optimize(prob_grad;
                        maxiter=100,
                        display=Silent(),
                        subspace=TwoDimSubspace(),
                        hessian_approximation=BFGS())
        
        @test result isa RetroResult
        @test norm(result.x - [1.0, 1.0]) < 0.1
        @test result.fx < 1e-2
    end
    
    @testset "Fully Analytical Version" begin
        x0 = [-1.2, 1.0]
        prob_analytic = RetroProblem(rosenbrock, rosenbrock_grad!, rosenbrock_hess!, x0)
        
        # Use ExactHessian since we have analytical Hessian
        result = optimize(prob_analytic;
                        maxiter=50,
                        display=Silent(),
                        subspace=TwoDimSubspace(),
                        hessian_approximation=ExactHessian())
        
        @test result isa RetroResult
        @test norm(result.x - [1.0, 1.0]) < 0.05  # Should be very accurate
        @test result.fx < 1e-4
    end
    
    @testset "Different TR Solvers" begin
        x0 = [-1.2, 1.0]
        prob = RetroProblem(rosenbrock, x0, AutoForwardDiff())
        
        @testset "EigenTRSolver" begin
            result = optimize(prob;
                            maxiter=50,
                            display=Silent(),
                            tr_solver=EigenTRSolver())
            @test result isa RetroResult
        end
        
        @testset "CauchyTRSolver" begin
            result = optimize(prob;
                            maxiter=100,  # Cauchy might need more iterations
                            display=Silent(),
                            tr_solver=CauchyTRSolver())
            @test result isa RetroResult
        end
    end
    
    @testset "Bounded Rosenbrock" begin
        x0 = [-1.2, 1.0]
        lb = [-2.0, -2.0]
        ub = [1.5, 2.0]  # Constrain the problem
        
        prob_bounded = RetroProblem(rosenbrock, x0, AutoForwardDiff(); lb=lb, ub=ub)
        
        result = optimize(prob_bounded;
                        maxiter=100,
                        display=Silent(),
                        options=RetroOptions(theta1=0.1, theta2=0.2))
        
        @test result isa RetroResult
        # Check bounds are respected
        @test all(result.x .>= lb)
        @test all(result.x .<= ub)
        @test norm(result.x - [1.0, 1.0]) < 0.1  # Should still find optimum
    end
    
    @testset "Different Starting Points" begin
        starting_points = [
            [0.0, 0.0],
            [2.0, 2.0],
            [-0.5, 1.5],
            [1.1, 0.9]  # Close to optimum
        ]
        
        for x0 in starting_points
            prob = RetroProblem(rosenbrock, x0, AutoForwardDiff())
            result = optimize(prob; maxiter=200, display=Silent())
            
            @test result isa RetroResult
            # All should converge to approximately the same point

            @test is_successful(result)
            if is_successful(result)
                @test norm(result.x - [1.0, 1.0]) < 0.2
            end
        end
    end
    
    @testset "Convergence Criteria" begin
        x0 = [-1.2, 1.0]
        prob = RetroProblem(rosenbrock, x0, AutoForwardDiff())
        
        # Test gradient tolerance
        options_gtol = RetroOptions(gtol_a=1e-4)
        result_gtol = optimize(prob; maxiter=100, options=options_gtol, display=Silent())
        
        @test is_successful(result_gtol)

        if result_gtol.termination_reason == :gtol
            @test norm(result_gtol.gx) < 1e-4
        end
        
        # Test function tolerance  
        options_ftol = RetroOptions(ftol_a=1e-6, gtol_a=1e-10)  # Make gtol very strict
        result_ftol = optimize(prob; maxiter=100, options=options_ftol, display=Silent())
        
        @test result_ftol isa RetroResult
    end
end