# ============================================================================
# Hessian Updates
# ============================================================================

function update_hessian!(state::TrustRegionState, ::BFGSUpdate, f, adtype)
    s = state.step_reflected
    y = state.g_trial - state.gx
    sy = dot(s, y)
    T = eltype(s)
    
    if sy > eps(T) * norm(s) * norm(y)
        Hs = state.Hx * s
        sHs = dot(s, Hs)
        
        if sHs > eps(T)
            state.Hx .-= (Hs * Hs') ./ sHs
            state.Hx .+= (y * y') ./ sy
        end
    end
end

function update_hessian!(state::TrustRegionState, ::SR1Update, f, adtype)
    s = state.step_reflected
    y = state.g_trial - state.gx
    Hs = state.Hx * s
    r = y - Hs
    rs = dot(r, s)
    T = eltype(s)
    
    if abs(rs) > eps(T) * norm(r) * norm(s)
        state.Hx .+= (r * r') ./ rs
    end
end

function update_hessian!(state::TrustRegionState, ::ExactHessian, f, adtype)
    hessian!(f, state.Hx, adtype, state.x)
    state.h_evals += 1
end
