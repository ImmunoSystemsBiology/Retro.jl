# Compile hot optimize paths during package precompilation to reduce
# time-to-first-result in fresh Julia sessions.
if ccall(:jl_generating_output, Cint, ()) == 1
    let
        x0 = [2.0, -1.0, 0.5, 1.5]
        lb = fill(-5.0, length(x0))
        ub = fill(5.0, length(x0))

        f(x) = begin
            s = 0.0
            @inbounds for i in eachindex(x)
                d = x[i] - 1.0
                s += d * d
            end
            return s
        end

        function g!(g, x)
            @inbounds for i in eachindex(x)
                g[i] = 2.0 * (x[i] - 1.0)
            end
            return g
        end

        function h!(H, x)
            fill!(H, 0.0)
            @inbounds for i in 1:axes(H,1)
                H[i, i] = 2.0
            end
            return H
        end

        prob_b = RetroProblem(f, g!, h!, x0; lb = lb, ub = ub)
        prob_u = RetroProblem(f, g!, h!, x0)

        optimize(prob_b; maxiter = 2, display = Silent(), hessian_approximation = ExactHessian(), subspace = TwoDimSubspace())
        optimize(prob_u; maxiter = 2, display = Silent(), hessian_approximation = ExactHessian(), subspace = TwoDimSubspace())

        prob_ad = RetroProblem(f, x0, AutoForwardDiff())
        optimize(prob_ad; maxiter = 2, display = Silent())
    end
end
