# ============================================================================
# Hessian Updates
# ============================================================================

function update_hessian!(state::TrustRegionState, ::BFGSUpdate, f, adtype)
    s = state.step_reflected

    # Compute gradient difference
    y = state.Δg
    y .= gx_trial(state) .- gx(state)
    sy = dot(s, y)
    T = eltype(s)
    
    if sy > eps(T) * norm(s) * norm(y)
        H = state.Hx_approx  # Access approximation directly
        mul!(state.Hs, H, s)
        sHs = dot(s, state.Hs)
        
        if sHs > eps(T)
            # In-place BFGS update
            BLAS.ger!(-one(T)/sHs, state.Hs, state.Hs, H)
            BLAS.ger!(one(T)/sy, y, y, H)
        end
    end
end

function update_hessian!(state::TrustRegionState, ::SR1Update, f, adtype)
    s = state.step_reflected
    
    # Compute gradient difference
    y = state.Δg
    y .= gx_trial(state) .- gx(state)
    
    H = state.Hx_approx  # Access approximation directly
    mul!(state.Hs, H, s)
    
    # Compute r = y - Hs in place (reuse Δg)
    r = state.Δg
    @. r = y - state.Hs
    
    rs = dot(r, s)
    T = eltype(s)
    
    if abs(rs) > eps(T) * norm(r) * norm(s)
        # In-place SR1 update
        BLAS.ger!(one(T)/rs, r, r, H)
    end
end

function update_hessian!(state::TrustRegionState, ::ExactHessian, f, adtype)
    # Compute exact Hessian using DiffResults
    hessian!(f, state.diff_result, adtype, state.x)
    state.h_evals += 1
end
