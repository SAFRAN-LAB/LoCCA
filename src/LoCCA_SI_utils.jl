




"""
    In this file we will develop the functions needed for the constructing the 
    low rank approximaiton of the kernel matrix using the neighborhood of the chebyshev nodes 
    from the source and target domains.
"""

using LinearAlgebra
using NearestNeighbors

function chev_nbd(domain_grid::AbstractMatrix{<:Real},
                      cheb_grid::AbstractMatrix{<:Real},
                      l::Int
)

    # Build tree only once
    tree = KDTree(permutedims(domain_grid))

    selected_mask = falses(size(domain_grid,1))
    selected_nodes = Matrix{eltype(domain_grid)}(undef, 0, size(domain_grid,2))

    pp = size(cheb_grid, 1)

    for i in 1 : pp

        center = cheb_grid[i,:]

        # Ask for extra neighbors
        k_extra = min(5l, size(domain_grid,1))

        idxs, _ = knn(tree, center, k_extra, true)

        # Remove already-used points
        idxs = filter(j -> !selected_mask[j], idxs)

        # Take first l available
        idxs = idxs[1:min(l, length(idxs))]

        # Mark selected
        selected_mask[idxs] .= true

        # Append
        selected_nodes = vcat(selected_nodes, domain_grid[idxs,:])
    end

    return selected_nodes 
end




# ---------------------------------------------------------
# Actual LoCCA with the compression.
# ---------------------------------------------------------
function LoCCA_with_grid_input(
    target_grid::AbstractMatrix{<:Real},
    source_grid::AbstractMatrix{<:Real},
    x_cheb_grid::AbstractMatrix{<:Real},
    y_cheb_grid::AbstractMatrix{<:Real},
    l::Int,
    kernel_type::String,
    tol:: Real = 1e-8
)

    # --------------------------------
    # Construct neighborhoods
    # --------------------------------
    X_CHEB_NBDs = chev_nbd(target_grid, x_cheb_grid, l)

    Y_CHEB_NBDs = chev_nbd(source_grid, y_cheb_grid, l)


    # --------------------------------
    # Get the kernel matrix for the neighborhood points
    # --------------------------------

    Kw = get_matrix_from_grid_nodes(X_CHEB_NBDs, Y_CHEB_NBDs, kernel_type)


    Fcol = qr(Kw, ColumnNorm())
    diagR_col = abs.(diag(Fcol.R))
    r_col = sum(diagR_col .> tol * diagR_col[1])

    piv_cols = Fcol.p[1:r_col]
    Yhat = Y_CHEB_NBDs[piv_cols, :]

    Frow = qr(Kw', ColumnNorm())
    diagR_row = abs.(diag(Frow.R))
    r_row = sum(diagR_row .> tol * diagR_row[1])

    piv_rows = Frow.p[1:r_row]
    Xhat = X_CHEB_NBDs[piv_rows, :]

    return Xhat, Yhat, r_row

end







# ---------------------------------------------------------
# Actual LoCCA with user-specified rank.
# ---------------------------------------------------------
function LoCCA_with_grid_input_rank(
    target_grid::AbstractMatrix{<:Real},
    source_grid::AbstractMatrix{<:Real},
    x_cheb_grid::AbstractMatrix{<:Real},
    y_cheb_grid::AbstractMatrix{<:Real},
    l::Int,
    kernel_type::String,
    r::Int
)

    # --------------------------------
    # Construct neighborhoods
    # --------------------------------
    X_CHEB_NBDs = chev_nbd(target_grid, x_cheb_grid, l)

    Y_CHEB_NBDs = chev_nbd(source_grid, y_cheb_grid, l)

    # --------------------------------
    # Kernel matrix on neighborhood points
    # --------------------------------
    Kw = get_matrix_from_grid_nodes(
        X_CHEB_NBDs,
        Y_CHEB_NBDs,
        kernel_type
    )

    # --------------------------------
    # Column pivots
    # --------------------------------
    Fcol = qr(Kw, ColumnNorm())

    r_col = min(r, length(Fcol.p))

    piv_cols = Fcol.p[1:r_col]
    Yhat = Y_CHEB_NBDs[piv_cols, :]

    # --------------------------------
    # Row pivots
    # --------------------------------
    Frow = qr(Kw', ColumnNorm())

    r_row = min(r, length(Frow.p))

    piv_rows = Frow.p[1:r_row]
    Xhat = X_CHEB_NBDs[piv_rows, :]

    return Xhat, Yhat, r_row
end