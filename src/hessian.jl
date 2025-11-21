"""
    Hessian Update Strategies

Implementations of various Hessian approximation update methods for
quasi-Newton optimization.
"""


"""
    update_hessian!(state, ::BFGSUpdate, f, adtype)

Update Hessian approximation using the BFGS formula.

BFGS (Broyden-Fletcher-Goldfarb-Shanno) is a rank-2 update that maintains
positive-definiteness and symmetry. It's the most robust quasi-Newton method
for most optimization problems.

The update satisfies the secant equation: H_new * s = y
where s is the step and y is the gradient difference.
"""
function update_hessian!(state::TrustRegionState, ::BFGSUpdate, f, adtype)
    s = state.step_reflected

    # Compute gradient difference
    y = state.Δg
    y .= gx_trial(state) .- gx(state)
    sy = dot(s, y)
    T = eltype(s)
    
    if sy > eps(T) * norm(s) * norm(y)
        H = state.Hx_approx
        mul!(state.Hs, H, s)
        sHs = dot(s, state.Hs)
        
        if sHs > eps(T)
            # In-place BFGS update: H_new = H - (Hs)(Hs)'/sHs + yy'/sy
            H .-= (state.Hs * state.Hs') ./ sHs
            H .+= (y * y') ./ sy
        end
    end
end

"""
    update_hessian!(state, ::SR1Update, f, adtype)

Update Hessian approximation using the SR1 (Symmetric Rank-1) formula.

SR1 can better capture indefinite Hessians than BFGS but may lose
positive-definiteness. The update is skipped if it would be too small
or degenerate.

The update satisfies: H_new * s = y
"""
function update_hessian!(state::TrustRegionState, ::SR1Update, f, adtype)
    s = state.step_reflected
    
    # Compute gradient difference
    y = state.Δg
    y .= gx_trial(state) .- gx(state)
    
    H = state.Hx_approx
    mul!(state.Hs, H, s)
    
    # Compute r = y - Hs
    r = state.Δg
    @. r = y - state.Hs
    
    rs = dot(r, s)
    T = eltype(s)
    
    if abs(rs) > eps(T) * norm(r) * norm(s)
        # In-place SR1 update: H_new = H + rr'/rs
        H .+= (r * r') ./ rs
    end
end

"""
    update_hessian!(state, ::ExactHessian, f, adtype)

Compute exact Hessian using automatic differentiation.

Computes the full Hessian matrix via second-order AD. Most accurate but
most expensive option. The Hessian is stored in the DiffResults buffer.
"""
function update_hessian!(state::TrustRegionState, ::ExactHessian, f, adtype)
    hessian!(f, state.diff_result, adtype, state.x)
    state.h_evals += 1
end
