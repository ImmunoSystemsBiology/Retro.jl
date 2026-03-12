using Retro
using Test
using LinearAlgebra
using Random

@testset "Challenging Optimization Problems" begin
    
    @testset "Extended Rosenbrock" begin
        # Extended Rosenbrock: sum over pairs of variables
        # f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
        function extended_rosenbrock(x)
            n = length(x)
            sum = 0.0
            for i in 1:n-1
                sum += 100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2
            end
            return sum
        end
        
        # Test with moderate dimension
        n = 10
        x0 = zeros(n)
        x0[1:2:end] .= -1.2  # Alternate starting points
        x0[2:2:end] .= 1.0
        
        prob = RetroProblem(extended_rosenbrock, x0, AutoForwardDiff())
        
        @testset "CG Subspace (Large Scale)" begin
            result = optimize(prob;
                            maxiter=100,
                            display=Silent(),
                            subspace=CGSubspace(max_cg_iter=min(n, 20)),
                            hessian_approximation=BFGS())
            
            @test result isa RetroResult
            @test result.iterations ≤ 100
            
            if is_successful(result)
                @test norm(result.x .- 1.0) < 0.5  # Should be close to all ones
            end
        end
    end
    
    @testset "Himmelblau's Function" begin
        # f(x,y) = (x² + y - 11)² + (x + y² - 7)²
        # Has 4 global minima: (3,2), (-2.8, 3.1), (-3.8, -3.3), (3.6, -1.8)
        himmelblau(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
        
        starting_points = [
            [0.0, 0.0],
            [1.0, 1.0], 
            [-1.0, -1.0],
            [5.0, 5.0],
            [-5.0, -5.0]
        ]
        
        found_minima = []
        
        for x0 in starting_points
            prob = RetroProblem(himmelblau, x0, AutoForwardDiff())
            result = optimize(prob; maxiter=100, display=Silent())
            
            @test result isa RetroResult
            
            if is_successful(result) && result.fx < 1e-6
                push!(found_minima, result.x)
            end
        end
        
        @test length(found_minima) ≥ 1  # Should find at least one minimum
    end
    
    @testset "Rastrigin Function (Many Local Minima)" begin
        # f(x) = A*n + ∑ᵢ[xᵢ² - A*cos(2π*xᵢ)]
        # Highly multimodal with many local minima
        function rastrigin(x::Vector{T}) where T
            A = 10.0
            n = length(x)
            return A * n + sum(xᵢ^2 - A * cos(2π * xᵢ) for xᵢ in x)
        end
        
        # Start close to global minimum
        x0 = [0.1, -0.1, 0.05]  # Close to [0,0,0]
        prob = RetroProblem(rastrigin, x0, AutoForwardDiff())
        
        result = optimize(prob;
                        maxiter=50,
                        display=Silent(),
                        options=RetroOptions(gtol_a=1e-4))
        
        @test result isa RetroResult
        # This is a very challenging function, so we're lenient
        if is_successful(result)
            @test norm(result.x) < 0.5  # Should be close to origin
        end
    end
    
    @testset "Beale Function (Narrow Valley)" begin
        # f(x,y) = (1.5 - x + xy)² + (2.25 - x + x*y²)² + (2.625 - x + x*y³)²
        # Global minimum at (3, 0.5)
        function beale(x)
            return (1.5 - x[1] + x[1]*x[2])^2 + 
                   (2.25 - x[1] + x[1]*x[2]^2)^2 + 
                   (2.625 - x[1] + x[1]*x[2]^3)^2
        end
        
        x0 = [1.0, 1.0]
        prob = RetroProblem(beale, x0, AutoForwardDiff())
        
        result = optimize(prob;
                        maxiter=100,
                        display=Silent(),
                        subspace=TwoDimSubspace())
        
        @test result isa RetroResult
        @test is_successful(result)
        if is_successful(result)
            @test norm(result.x - [3.0, 0.5]) < 0.1
            @test result.fx < 1e-6
        end
    end
    
    @testset "Booth Function" begin
        # f(x,y) = (x + 2y - 7)² + (2x + y - 5)²
        # Global minimum at (1, 3) with value 0
        booth(x) = (x[1] + 2*x[2] - 7)^2 + (2*x[1] + x[2] - 5)^2
        
        x0 = [0.0, 0.0]
        prob = RetroProblem(booth, x0, AutoForwardDiff())
        
        result = optimize(prob; maxiter=50, display=Silent())
        
        @test result isa RetroResult
        @test is_successful(result)
        if is_successful(result)
            @test norm(result.x - [1.0, 3.0]) < 1e-4
            @test result.fx < 1e-8
        end
    end
    
    @testset "Powell Badly Scaled Function" begin
        # f(x,y) = (10000*x*y - 1)² + (exp(-x) + exp(-y) - 1.0001)²
        # Very badly scaled, challenging for numerical methods
        function powell_badly_scaled(x)
            return (10000*x[1]*x[2] - 1)^2 + (exp(-x[1]) + exp(-x[2]) - 1.0001)^2
        end
        
        x0 = [0.0, 1.0]  # Standard starting point
        prob = RetroProblem(powell_badly_scaled, x0, AutoForwardDiff())
        
        result = optimize(prob;
                        maxiter=100,
                        display=Silent(),
                        options=RetroOptions(gtol_a=1e-5))  # Relaxed tolerance
        
        @test result isa RetroResult
        @test result.fx > 0  # Should be finite
        @test is_successful(result)
    end
    
    @testset "Sphere Function with Noise" begin
        # Add noise to simple sphere function to test robustness
        Random.seed!(1234)
        noise_scale = 1e-8
        
        function noisy_sphere(x)
            base_val = sum(abs2, x)
            noise = noise_scale * randn()
            return base_val + noise
        end
        
        x0 = [1.0, 1.0]
        prob = RetroProblem(noisy_sphere, x0, AutoForwardDiff())
        
        result = optimize(prob;
                        maxiter=50,
                        display=Silent(),
                        options=RetroOptions(gtol_a=1e-4))
        
        @test result isa RetroResult
        # Should still find the minimum despite noise
        if is_successful(result)
            @test norm(result.x) < 0.1
        end
    end
    
    @testset "Constrained Problems (Bounds)" begin
        @testset "Box-Constrained Rosenbrock" begin
            # Rosenbrock with tight bounds that exclude the unconstrained minimum
            rosenbrock(x) = 100*(x[2] - x[1]^2)^2 + (1 - x[1])^2
            
            x0 = [0.0, 0.0]
            lb = [-0.5, -0.5]  # Bounds that exclude (1,1)
            ub = [0.8, 0.8]
            
            prob = RetroProblem(rosenbrock, x0, AutoForwardDiff(); lb=lb, ub=ub)
            
            result = optimize(prob;
                            maxiter=100,
                            display=Silent(),
                            options=RetroOptions(theta1=0.05, theta2=0.1))
            
            @test result isa RetroResult
            # Check bounds are respected
            @test all(result.x .>= lb .- 1e-10)
            @test all(result.x .<= ub .+ 1e-10)
            
            # Should find constrained minimum near boundary
            if is_successful(result)
                @test abs(result.x[1] - 0.8) < 0.1 || abs(result.x[1] - (-0.5)) < 0.1
            end
        end
        
        @testset "One-Sided Bounds" begin
            # Simple quadratic with lower bounds only
            f(x) = sum(abs2, x .- [-2.0, -3.0])  # Minimum at (-2, -3)
            
            x0 = [0.0, 0.0]
            lb = [-1.0, -1.0]  # Prevent reaching true minimum
            ub = [Inf, Inf]
            
            prob = RetroProblem(f, x0, AutoForwardDiff(); lb=lb, ub=ub)
            
            result = optimize(prob; maxiter=50, display=Silent())
            
            @test result isa RetroResult
            @test all(result.x .>= lb .- 1e-10)
            
            if is_successful(result)
                # Should be at boundary
                @test abs(result.x[1] - (-1.0)) < 1e-6
                @test abs(result.x[2] - (-1.0)) < 1e-6
            end
        end
    end
    
    @testset "Ill-Conditioned Problems" begin
        @testset "Extreme Conditioning" begin
            # f(x,y) = 100000*x² + y²
            # Very different scales in different dimensions
            extreme_quad(x) = 100000 * x[1]^2 + x[2]^2
            
            x0 = [1.0, 1.0]
            prob = RetroProblem(extreme_quad, x0, AutoForwardDiff())
            
            result = optimize(prob;
                            maxiter=100,
                            display=Silent(),
                            subspace=TwoDimSubspace())
            
            @test result isa RetroResult
            
            if is_successful(result)
                @test abs(result.x[1]) < 1e-3  # Should find minimum accurately
                @test abs(result.x[2]) < 1e-3
            end
        end
    end
end