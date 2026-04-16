"""
    Silent <: AbstractDisplayMode

Suppress all output during optimization.
"""
struct Silent <: AbstractDisplayMode end

"""
    Iteration <: AbstractDisplayMode

Print a status line after every iteration.
"""
struct Iteration <: AbstractDisplayMode end

"""
    Final <: AbstractDisplayMode

Print only a summary after the optimizer terminates.
"""
struct Final <: AbstractDisplayMode end

"""
    Verbose <: AbstractDisplayMode

Print per-iteration output plus a progress bar (via ProgressMeter).
"""
struct Verbose <: AbstractDisplayMode end

"""
    Debug <: AbstractDisplayMode

Print exhaustive diagnostic output for every iteration.

Shows the current point, gradient, Hessian diagonal/condition, step details,
predicted vs. actual reduction, ρ, trust-region radius update, and
accept/reject reasoning.  Useful for investigating why steps are rejected
in problems like ODE parameter estimation.
"""
struct Debug <: AbstractDisplayMode end

function display_header(::Silent) end

function display_header(::Union{Iteration, Final, Verbose})
    @printf "%-6s %-12s %-12s %-12s %-12s %-10s\n" "Iter" "f(x)" "||g||" "Δ" "ρ" "Status"
    @printf "%s\n" repeat("-", 70)
end

function display_iteration(::Silent, iter, f, g_norm, Delta, rho, status) end

function display_iteration(::Union{Iteration, Verbose}, iter, f, g_norm, Delta, rho, status)
    @printf "%-6d %-12.6e %-12.6e %-12.6e %-12.4f %-10s\n" iter f g_norm Delta rho status
end

function display_iteration(::Final, iter, f, g_norm, Delta, rho, status)
    # Only display at the end
end

function display_final(::Silent, result) end

function display_final(::Union{Final, Iteration, Verbose}, result)
    println()
    @printf "Optimization completed:\n"
    @printf "  Final objective value: %.6e\n" result.fx
    @printf "  Final gradient norm:   %.6e\n" norm(result.gx)
    @printf "  Iterations:            %d\n" result.iterations
    @printf "  Function evaluations:  %d\n" result.function_evaluations
    @printf "  Gradient evaluations:  %d\n" result.gradient_evaluations
    @printf "  Termination reason:    %s\n" result.termination_reason
end

function display_header(::Debug)
    println("═"^72)
    println("  RETRO DEBUG MODE — Detailed iteration diagnostics")
    println("═"^72)
end

display_iteration(::Debug, iter, f, g_norm, Delta, rho, status) = nothing

function display_final(::Debug, result)
    println()
    println("═"^72)
    @printf "  DEBUG SUMMARY\n"
    @printf "  Final objective value: %.10e\n" result.fx
    @printf "  Final gradient norm:   %.10e\n" norm(result.gx)
    @printf "  Iterations:            %d\n" result.iterations
    @printf "  Function evaluations:  %d\n" result.function_evaluations
    @printf "  Gradient evaluations:  %d\n" result.gradient_evaluations
    @printf "  Termination reason:    %s\n" result.termination_reason
    println("═"^72)
end

display_debug_initial(::AbstractDisplayMode, args...) = nothing

function display_debug_initial(::Debug, x, f, g, g_norm, Delta)
    println()
    println("┌─── Initial State ", "─"^(72 - 19))
    println("│")
    _debug_print_vec("│   x₀", x)
    @printf "│   f(x₀)    = %.10e\n" f
    @printf "│   ‖g₀‖     = %.10e\n" g_norm
    _debug_print_vec("│   g₀", g)
    @printf "│   Δ₀       = %.10e\n" Delta
    println("│")
    println("└", "─"^71)
end

display_debug_info(::AbstractDisplayMode, args...; kwargs...) = nothing

function display_debug_info(
    ::Debug, k, x, g, g_norm, p, x_trial, Hp,
    step_norm, f_current, f_trial,
    pred_red, actual_red, rho,
    Delta_old, Delta_new, status,
    consecutive_rejections, mu,
    g_dot_p, p_dot_Hp
)
    hdr = "─── Iteration $k "
    println()
    println("┌", hdr, "─"^max(0, 71 - length(hdr) - 1))
    println("│")

    println("│ State:")
    _debug_print_vec("│   x", x)
    @printf "│   f(x)       = %.10e\n" f_current
    @printf "│   ‖g‖        = %.10e\n" g_norm
    _debug_print_vec("│   g", g)
    @printf "│   Δ          = %.10e\n" Delta_old
    println("│")

    println("│ Step:")
    _debug_print_vec("│   p", p)
    _debug_print_vec("│   x_trial", x_trial)
    @printf "│   ‖p‖        = %.10e\n" step_norm
    _debug_print_vec("│   Hp", Hp)
    @printf "│   gᵀp        = %.10e\n" g_dot_p
    @printf "│   pᵀHp       = %.10e\n" p_dot_Hp
    println("│")

    println("│ Reduction:")
    @printf "│   pred_red    = %.10e   (−gᵀp − ½ pᵀHp)\n" pred_red
    @printf "│   actual_red  = %.10e   (f − f_trial)\n" actual_red
    @printf "│   f_trial     = %.10e\n" f_trial
    @printf "│   ρ           = %+.10f\n" rho

    if pred_red ≤ 0
        println("│   ⚠  Predicted reduction ≤ 0 — no improvement!")
    end
    if actual_red < 0
        println("│   ⚠  Actual reduction < 0 — objective INCREASED by $(@sprintf("%.6e", -actual_red))")
    end
    println("│")

    println("│ Decision: ", status)
    @printf "│   ρ = %+.8f   vs   μ = %.4f\n" rho mu

    if Delta_new < Delta_old
        @printf "│   Δ: %.6e → %.6e  (shrunk ×%.4f)\n" Delta_old Delta_new (Delta_new / Delta_old)
    elseif Delta_new > Delta_old
        @printf "│   Δ: %.6e → %.6e  (expanded ×%.4f)\n" Delta_old Delta_new (Delta_new / Delta_old)
    else
        @printf "│   Δ: %.6e  (unchanged)\n" Delta_old
    end

    if consecutive_rejections > 0
        @printf "│   Consecutive rejections: %d\n" consecutive_rejections
    end

    println("└", "─"^71)
end

function _debug_print_vec(prefix, v)
    n = length(v)
    if n ≤ 8
        vals = join([@sprintf("%.6e", vi) for vi in v], ", ")
        println("$prefix = [$vals]")
    else
        first4 = join([@sprintf("%.6e", v[i]) for i in 1:4], ", ")
        last2  = join([@sprintf("%.6e", v[i]) for i in n-1:n], ", ")
        println("$prefix = [$first4, …, $last2]  (n=$n)")
    end
end

mutable struct RetroProgress
    meter::Union{Progress, Nothing}
    display_mode::AbstractDisplayMode
    last_update::Int
    
    function RetroProgress(maxiter::Int, display_mode::AbstractDisplayMode)
        if isa(display_mode, Verbose)
            meter = Progress(maxiter, desc="Optimizing: ", showspeed=true)
        else
            meter = nothing
        end
        new(meter, display_mode, 0)
    end
end

function update_progress!(progress::RetroProgress, iter::Int, f_val, g_norm, info::String="")
    if progress.meter !== nothing
        if iter > progress.last_update
            ProgressMeter.update!(progress.meter, iter, showvalues=[
                ("f(x)", f_val),
                ("||∇f(x)||", g_norm),
                ("Info", info)
            ])
            progress.last_update = iter
        end
    end
end

function finish_progress!(progress::RetroProgress)
    if progress.meter !== nothing
        ProgressMeter.finish!(progress.meter)
    end
end

