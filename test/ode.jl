using Retro
using Test
using OrdinaryDiffEq
using Random
using LinearAlgebra

@testset "ODE Parameter Estimation Problems" begin

    @testset "SIR Model Parameter Estimation" begin
        # SIR model definition

        function sir_model!(du, u, p, t)
            β, γ = p
            S, I, R = u
            dS = -β * S * I
            dI = β * S * I - γ * I
            dR = γ * I
            du[1] = dS
            du[2] = dI
            du[3] = dR
        end

        # initial conditions and parameters
        u0 = [999.0, 1.0, 0.0]  # initial susceptible, infected, recovered
        p = [0.0005, 0.1]    # transmission rate β and recovery rate γ
        tspan = (0.0, 100.0)  # time span for simulation (days)

        # solve the ODE
        prob = ODEProblem(sir_model!, u0, tspan, p)

        function loss_function(p, N, initial_infected, new_infections, timepoints, u0, tspan)
            u0 = [N - initial_infected, initial_infected, 0.0]
            prob = ODEProblem(sir_model!, u0, tspan, exp.(p[1:end-1]))  # Exponentiate parameters to ensure positivity
            sol = solve(prob, saveat=timepoints)
            model_new_infections = -diff(sol[1, :])  # Change in S between consecutive timepoints
            n = length(new_infections)
            sigma = p[end]  # Last parameter is noise std dev
            nll = n/2 * log(2π) + n/2 * log(sigma^2) + sum(abs2, new_infections - model_new_infections) / (2 * sigma^2)
            return nll
        end

        # Generate the data (seeded for reproducibility)
        Random.seed!(42)
        u0 = [999.0, 1.0, 0.0]
        p_true = [0.0005, 0.1]
        sigma_true = 0.5
        tspan = (0.0, 100.0)
        timepoints = collect(0.0:2.0:100.0)

        prob = ODEProblem(sir_model!, u0, tspan, p_true)
        sol = solve(prob, saveat=timepoints)
        new_infections_true = -diff(sol[1, :])  # Change in S between consecutive timepoints
        new_infections_noisy = max.(0, new_infections_true .+ randn(length(new_infections_true)) .* sigma_true)  # Add noise and ensure non-negativity

        # Create the optimization problem
        N = 1000.0
        initial_infected = 1.0
        p_init = [log(0.0003), log(0.05), 1.0]  # Initial guess for parameters (including noise std dev)
        lb = [-10.0, -10.0, -10.0]; ub = [1.0, 1.0, 10.0]
        prob_opt = RetroProblem(p -> loss_function(p, N, initial_infected, new_infections_noisy, timepoints, u0, tspan), p_init, AutoForwardDiff(), lb=lb, ub=ub)

        # Optimize parameters — use relaxed gradient tolerance for noisy ODE objective
        opts = RetroOptions(gtol_a=1e-3, ftol_a=1e-3)
        result = optimize(prob_opt; maxiter=200, display=Silent(), options=opts)

        @test result isa RetroResult
        @test is_successful(result)
        @test abs(exp(result.x[1]) - p_true[1]) / p_true[1] < 0.05  # Estimated β within 5% of true value
        @test abs(exp(result.x[2]) - p_true[2]) / p_true[2] < 0.05  # Estimated γ within 5% of true value
        @test abs(result.x[3] - 0.5) / 0.5 < 0.1  # Estimated noise std dev within 10% of true value
    end
end