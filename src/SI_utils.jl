# SI_utils.jl

# ```
# this file implements the skeletonized interpolation (SI) method for low-rank approximation of kernel matrices. 
# The SI method is based on the idea of selecting a subset of rows and columns from the kernel matrix that captures 
# the essential information, allowing for efficient computation and storage.
# ```


using LowRankApprox            # the LowRankApprox.jl package is used to perform the strong RRQR
using LinearAlgebra            # for linear algebra operations



function skeletonized_interpolation_nodes(
    kernel_choice::String,
    target_domain::AbstractArray{Float64, 2},
    source_domain::AbstractArray{Float64, 2},
    p::Int;
    tol::Float64 = 1e-8
)

    Xbar = generate_chebyshev_grid(target_domain, p)
    Ybar = generate_chebyshev_grid(source_domain, p)

    K = get_matrix_from_grid_nodes(Xbar, Ybar, kernel_choice)

    wx = chebyshev_tensor_weights(target_domain, p)
    wy = chebyshev_tensor_weights(source_domain, p)

    sqrt_wx = sqrt.(wx)
    sqrt_wy = sqrt.(wy)

    Kw = sqrt_wx .* K .* sqrt_wy'

    Fcol = qr(Kw, ColumnNorm())
    diagR_col = abs.(diag(Fcol.R))
    r_col = sum(diagR_col .> tol * diagR_col[1])

    piv_cols = Fcol.p[1:r_col]
    Yhat = Ybar[piv_cols, :]

    Frow = qr(Kw', ColumnNorm())
    diagR_row = abs.(diag(Frow.R))
    r_row = sum(diagR_row .> tol * diagR_row[1])

    piv_rows = Frow.p[1:r_row]
    Xhat = Xbar[piv_rows, :]

    return Xhat, Yhat, piv_rows, piv_cols, Kw
end



function skeletonized_interpolation_nodes_rank(
    kernel_choice::String,
    target_domain::Matrix{Float64},
    source_domain::Matrix{Float64},
    p::Int,
    r::Int
)

    Xbar = generate_chebyshev_grid(target_domain, p)
    Ybar = generate_chebyshev_grid(source_domain, p)

    K = get_matrix_from_grid_nodes(Xbar, Ybar, kernel_choice)

    wx = chebyshev_tensor_weights(target_domain, p)
    wy = chebyshev_tensor_weights(source_domain, p)

    sqrt_wx = sqrt.(wx)
    sqrt_wy = sqrt.(wy)

    Kw = sqrt_wx .* K .* sqrt_wy'

    max_rank = min(size(Kw)...)
    @assert 1 <= r <= max_rank "Requested rank exceeds matrix dimensions"

    Fcol = qr(Kw, ColumnNorm())
    piv_cols = Fcol.p[1:r]
    Yhat = Ybar[piv_cols, :]

    Frow = qr(Kw', ColumnNorm())
    piv_rows = Frow.p[1:r]
    Xhat = Xbar[piv_rows, :]

    return Xhat, Yhat, piv_rows, piv_cols, Kw
end

#--------------
# now we try to perform si to the arc and dot domain for that we pass the chebyshev grid into the fuction and 
# get the skeleton nodes for each sub-box and then merge them to get the unified global skeleton nodes for the 
# entire arc and dot domain. We can also remove any exact duplicate rows if there are any coordinates that overlap 
# perfectly between the sub-boxes.  
# -------------  


function SI_nodes_from_grids(
    kernel_choice::String,
    Xbar::Matrix{Float64},
    Ybar::Matrix{Float64},
    wx::Vector{Float64},
    wy::Vector{Float64},
    r::Int
)

    K = get_matrix_from_grid_nodes(Xbar, Ybar, kernel_choice)

    sqrt_wx = sqrt.(wx)
    sqrt_wy = sqrt.(wy)

    Kw = sqrt_wx .* K .* sqrt_wy'

    max_rank = min(size(Kw)...)
    @assert 1 <= r <= max_rank "Requested rank exceeds matrix dimensions"

    Fcol = qr(Kw, ColumnNorm())
    piv_cols = Fcol.p[1:r]
    Yhat = Ybar[piv_cols, :]

    Frow = qr(Kw', ColumnNorm())
    piv_rows = Frow.p[1:r]
    Xhat = Xbar[piv_rows, :]

    return Xhat, Yhat, piv_rows, piv_cols, Kw
end





function SI_nodes_from_grids_SRRQR(
    kernel_choice::String,
    Xbar::Matrix{Float64},
    Ybar::Matrix{Float64},
    wx::Vector{Float64},
    wy::Vector{Float64},
    r::Int
)
    K = get_matrix_from_grid_nodes(Xbar, Ybar, kernel_choice)

    sqrt_wx = sqrt.(wx)
    sqrt_wy = sqrt.(wy)

    Kw = sqrt_wx .* K .* sqrt_wy'

    max_rank = min(size(Kw)...)
    @assert 1 <= r <= max_rank "Requested rank exceeds matrix dimensions"

    # 1. Column-pivoting using Strong RRQR (pqrfact uses SRRQR by default)
    Fcol = pqrfact(Kw, rank=r)
    piv_cols = Fcol.p[1:r]
    Yhat = Ybar[piv_cols, :]

    # 2. Row-pivoting using Strong RRQR on the transpose
    Frow = pqrfact(Kw', rank=r)
    piv_rows = Frow.p[1:r]
    Xhat = Xbar[piv_rows, :]

    return Xhat, Yhat, piv_rows, piv_cols, Kw
end
