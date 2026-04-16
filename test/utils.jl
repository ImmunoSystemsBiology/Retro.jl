using Retro
using Test
using LinearAlgebra

@testset "Norm Utilities" begin
    @testset "norm_inf" begin
        x = [1.0, -3.0, 2.0]
        @test norm_inf(x) ≈ 3.0
        @test norm_inf([0.0]) ≈ 0.0
        @test norm_inf([-5.0, 1.0]) ≈ 5.0
    end

    @testset "norm_weighted" begin
        x = [1.0, 2.0]
        w = [1.0, 1.0]
            @test norm_weighted(x, w) ≈ sqrt(5.0)
        w2 = [2.0, 3.0]
            @test norm_weighted(x, w2) ≈ sqrt((1.0*2.0)^2 + (2.0*3.0)^2)
        # Dimension mismatch
            @test_throws DimensionMismatch norm_weighted([1.0, 2.0], [1.0])
    end

    @testset "norm_relative" begin
        x = [3.0, 4.0]
        x_ref = [0.0, 0.0]
            @test norm_relative(x, x_ref) ≈ 5.0 / (1.0 + 0.0)
        x_ref2 = [3.0, 4.0]
            @test norm_relative(x, x_ref2) ≈ 5.0 / (1.0 + 5.0)
    end

    @testset "norm_scaled" begin
        x = [2.0, 3.0]
        s = [2.0, 3.0]
            @test norm_scaled(x, s) ≈ sqrt(2.0)  # sqrt(1^2 + 1^2)
            @test_throws DimensionMismatch norm_scaled([1.0, 2.0], [1.0])
    end

    @testset "trust_region_norm" begin
        x  = [1.0, 1.0, 1.0]
        lb = [0.0, -Inf, -Inf]
        ub = [2.0, Inf, Inf]
            result = trust_region_norm(x, lb, ub)
        @test result > 0.0
        @test isfinite(result)
    end

    @testset "gradient_norm" begin
        g = [3.0, 4.0]
        x = [0.0, 0.0]
        # char_scale = max(norm_inf(x), 1) = 1
            @test gradient_norm(g, x) ≈ 5.0
        x2 = [10.0, 0.0]
        # char_scale = max(10, 1) = 10
            @test gradient_norm(g, x2) ≈ 5.0 / 10.0
    end
end

@testset "LinearAlgebra Helpers" begin
    @testset "safe_norm" begin
        x = [3.0, 4.0]
            @test safe_norm(x) ≈ 5.0
        # Inf values → should return 0
            @test safe_norm([Inf, 1.0]) == 0.0
    end

    @testset "safe_dot" begin
        x = [1.0, 2.0]
        y = [3.0, 4.0]
            @test safe_dot(x, y) ≈ 11.0
            @test safe_dot([Inf, 1.0], [1.0, 1.0]) == 0.0
    end

    @testset "safe_cond" begin
        A = [1.0 0.0; 0.0 2.0]
            @test isfinite(safe_cond(A))
            @test safe_cond(A) ≈ 2.0
    end

    @testset "is_positive_definite" begin
        A_pd = [2.0 0.0; 0.0 2.0]
            @test is_positive_definite(A_pd) == true
        A_not_pd = [-1.0 0.0; 0.0 2.0]
            @test is_positive_definite(A_not_pd) == false
    end

    @testset "smallest_eigenvalue" begin
        A = [3.0 0.0; 0.0 1.0]
            @test smallest_eigenvalue(A) ≈ 1.0
    end

    @testset "safe_copy!" begin
        src = [1.0, 2.0, 3.0]
        dest = zeros(3)
        Retro.safe_copy!(dest, src)
        @test dest == src
        @test_throws DimensionMismatch Retro.safe_copy!(zeros(2), src)
    end

    @testset "safe_axpy!" begin
        x = [1.0, 2.0]
        y = [3.0, 4.0]
        Retro.safe_axpy!(2.0, x, y)
        @test y ≈ [5.0, 8.0]
        @test_throws DimensionMismatch Retro.safe_axpy!(1.0, [1.0], [1.0, 2.0])
    end

    @testset "safe_scale!" begin
        x = [2.0, 4.0]
        Retro.safe_scale!(0.5, x)
        @test x ≈ [1.0, 2.0]
    end

    @testset "safe_fill!" begin
        x = zeros(3)
        Retro.safe_fill!(x, 7.0)
        @test x ≈ [7.0, 7.0, 7.0]
    end

    @testset "has_invalid_values" begin
        @test Retro.has_invalid_values([1.0, Inf]) == true
        @test Retro.has_invalid_values([1.0, NaN]) == true
        @test Retro.has_invalid_values([1.0, 2.0]) == false
    end

    @testset "clamp_vector!" begin
        x = [-2.0, 0.5, 3.0]
            Retro.clamp_vector!(x, -1.0, 1.0)
        @test x ≈ [-1.0, 0.5, 1.0]
    end
end
