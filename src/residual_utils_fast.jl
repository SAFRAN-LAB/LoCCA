# ============================================================
# residual_utils_fast.jl
# Residual computation for ND kernel approximation
# ============================================================

using LinearAlgebra



function precompute_residual_data(X::Matrix,
                                  Y::Matrix,
                                  w_x::Matrix,
                                  w_y::Matrix,
                                  K_k::Function)

    # --------------------------------------------------------
    # Generate tensor-product grids
    # --------------------------------------------------------
    X_grid = generate_grid_points(X)
    Y_grid = generate_grid_points(Y)

    # --------------------------------------------------------
    # Precompute full kernel matrix
    # --------------------------------------------------------
    Kxy = K_k(X_grid, Y_grid)

    # --------------------------------------------------------
    # Build weight vectors
    # --------------------------------------------------------
    W_x = make_weight_vec(w_x)
    W_y = make_weight_vec(w_y)

    Nq = size(Kxy,1)

    # --------------------------------------------------------
    # Precompute constant term
    # C = ∑ Wx(i) Wy(j) K(x_i,y_j)^2
    # --------------------------------------------------------
    C = 0.0

    @inbounds for j in 1:Nq
        wy = W_y[j]

        for i in 1:Nq
            C += W_x[i] * wy * Kxy[i,j]^2
        end
    end

    return (
        X_grid = X_grid,
        Y_grid = Y_grid,
        Kxy    = Kxy,
        W_x    = W_x,
        W_y    = W_y,
        C      = C,
        Nq     = Nq
    )
end





function get_residual_ND(t::AbstractVector,
                         s::AbstractVector,
                         K_k::Function,
                         pre)

    Nq = pre.Nq

    # --------------------------------------------------------
    # Evaluate K(x,s) and K(t,y)
    # --------------------------------------------------------
    Kxs = vec(K_k(pre.X_grid, reshape(s,1,:)))
    Kty = vec(K_k(reshape(t,1,:), pre.Y_grid))

    # Pivot
    Kts = K_k(reshape(t,1,:), reshape(s,1,:))[1,1]

    Wx = pre.W_x
    Wy = pre.W_y

    # --------------------------------------------------------
    # Term A = ∑ Wx(i) K(x_i,s)^2
    # --------------------------------------------------------
    Ax = 0.0

    @inbounds @simd for i in 1:Nq
        Ax += Wx[i] * Kxs[i]^2
    end

    # --------------------------------------------------------
    # Term B = ∑ Wy(j) K(t,y_j)^2
    # --------------------------------------------------------
    By = 0.0

    @inbounds @simd for j in 1:Nq
        By += Wy[j] * Kty[j]^2
    end

    # --------------------------------------------------------
    # Compute cross term
    # --------------------------------------------------------
    cross = 0.0

    @inbounds for j in 1:Nq

        vj = 0.0

        for i in 1:Nq
            vj += Wx[i] * pre.Kxy[i,j] * Kxs[i]
        end

        cross += Wy[j] * Kty[j] * vj
    end

    # --------------------------------------------------------
    # Final residual
    # --------------------------------------------------------
    res = pre.C -
          2 * cross / Kts +
          (Ax * By) / (Kts^2)

    return res #sqrt(res)
end




# ============================================================
# Construct Kronecker weight vector
#
# For 2D:
#   kron(W[2,:], W[1,:])
#
# For 3D:
#   kron(kron(W[3,:], W[2,:]), W[1,:])
# ============================================================
function make_weight_vec(W::Matrix)

    dim = size(W, 1)

    if dim == 2
        return kron(W[2, :], W[1, :])
    elseif dim == 3
        return kron(kron(W[3, :], W[2, :]), W[1, :])
    else
        error("Only 2D or 3D supported.")
    end
end