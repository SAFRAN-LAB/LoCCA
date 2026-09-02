




"""
    In this file we will develop the functions needed for the constructing the 
    low rank approximaiton of the kernel matrix using the neighborhood of the chebyshev nodes 
    from the source and target domains.
"""

using LinearAlgebra
using NearestNeighbors

function chev_nbd_knn(domain_grid::AbstractMatrix{<:Real},
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
# Low-rank approximation from CUR
# ---------------------------------------------------------
function cheb_nbd_lowrank_from_construction(
    target_grid::AbstractMatrix{<:Real},
    source_grid::AbstractMatrix{<:Real},
    x_cheb_grid::AbstractMatrix{<:Real},
    y_cheb_grid::AbstractMatrix{<:Real},
    l::Int,
    r::Int,
    kernel_type::String
)

    # --------------------------------
    # Construct neighborhoods
    # --------------------------------
    X_CHEB_NBDs = chev_nbd_knn(target_grid, x_cheb_grid, l)

    Y_CHEB_NBDs = chev_nbd_knn(source_grid, y_cheb_grid, l)

    # println("Size of X_CHEB_NBDs = ", size(X_CHEB_NBDs))

    # println("Size of Y_CHEB_NBDs = ", size(Y_CHEB_NBDs))

    # --------------------------------
    # Construct CUR matrices
    # --------------------------------
    C = get_matrix_from_grid_nodes(target_grid, Y_CHEB_NBDs, kernel_type)

    U = get_matrix_from_grid_nodes(X_CHEB_NBDs, Y_CHEB_NBDs, kernel_type)

    R = get_matrix_from_grid_nodes(X_CHEB_NBDs, source_grid, kernel_type)

    # --------------------------------
    # Economy QR
    # --------------------------------
    Qc, Rc = qr(C)

    Qr, Rr = qr(transpose(R))

    # convert QR factors to matrices
    Qc = Matrix(Qc)
    Rc = Matrix(Rc)

    Qr = Matrix(Qr)
    Rr = Matrix(Rr)

    # --------------------------------
    # Small core matrix
    # S = Rc * (U \ Rr')
    # --------------------------------
    X = U \ transpose(Rr)

    S = Rc * X

    # --------------------------------
    # Truncated SVD
    # --------------------------------
    Us, Ss, Vs = svd(S)

    Ur = Us[:,1:r]
    Sr = Ss[1:r]

    Vr = Vs[:,1:r]

    # --------------------------------
    # Lift back
    # --------------------------------
    Utilde = Qc * Ur

    Vtilde = Qr * Vr

    return Utilde, Diagonal(Sr), Vtilde
end




function cheb_nbd_lowrank_from_construction_tol(
    target_grid::AbstractMatrix{<:Real},
    source_grid::AbstractMatrix{<:Real},
    x_cheb_grid::AbstractMatrix{<:Real},
    y_cheb_grid::AbstractMatrix{<:Real},
    l::Int,
    tol::Real,
    kernel_type::String
)

    # 1. Construct neighborhoods
    X_CHEB_NBDs = chev_nbd_knn(target_grid, x_cheb_grid, l)
    Y_CHEB_NBDs = chev_nbd_knn(source_grid, y_cheb_grid, l)

    # 2. Construct CUR matrices
    C = get_matrix_from_grid_nodes(target_grid, Y_CHEB_NBDs, kernel_type)
    U = get_matrix_from_grid_nodes(X_CHEB_NBDs, Y_CHEB_NBDs, kernel_type)
    R = get_matrix_from_grid_nodes(X_CHEB_NBDs, source_grid, kernel_type)

    # 3. Economy QR (Using pivoted QR if matrices are ill-conditioned, 
    # but sticking to standard QR to match your exact behavior)
    Fc = qr(C)
    # Avoid Matrix(Fc.Q). Fc.R is already an upper triangular property, 
    # but we extract it safely using the property helper.
    Rc = Fc.R 

    Fr = qr(R') # Using lazy adjoint/transpose
    Rr = Fr.R

    # 4. Small core matrix: S = Rc * (U \ Rr')
    # Instead of computing U \ Rr', we can pre-factorize U if it's reused,
    # or just solve it directly. Rr' is a lazy adjoint.
    X = U \ Rr'
    S = Rc * X

    # 5. SVD of small core
    Us, singular_vals, Vs = svd(S)

    # 6. Determine numerical rank
    max_sv = singular_vals[1]
    # Fast count using a generator to avoid allocations
    r_eff = count(sv -> sv >= tol * max_sv, singular_vals)

    # 7. Truncate & Lift back simultaneously to avoid creating intermediate Ur/Vr
    # Fc.Q and Fr.Q act like matrices dynamically without allocating the full dense grid
    Utilde = Fc.Q * @view(Us[:, 1:r_eff])
    Vtilde = Fr.Q * @view(Vs[:, 1:r_eff])

    return Utilde, Diagonal(singular_vals[1:r_eff]), Vtilde, r_eff
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
    kernel_type::String
)

    # --------------------------------
    # Construct neighborhoods
    # --------------------------------
    X_CHEB_NBDs = chev_nbd_knn(target_grid, x_cheb_grid, l)

    Y_CHEB_NBDs = chev_nbd_knn(source_grid, y_cheb_grid, l)

    # println("Size of X_CHEB_NBDs = ", size(X_CHEB_NBDs))

    # println( "Size of Y_CHEB_NBDs = ", size(Y_CHEB_NBDs))

    # --------------------------------
    # Construct CUR matrices
    # --------------------------------
    C = get_matrix_from_grid_nodes(target_grid, Y_CHEB_NBDs, kernel_type)

    U = get_matrix_from_grid_nodes(X_CHEB_NBDs, Y_CHEB_NBDs, kernel_type)

    R = get_matrix_from_grid_nodes(X_CHEB_NBDs, source_grid, kernel_type)

    

    return C, U, R, size(U, 2) # rank is at most the number of columns in U
end

















# """
#     Optimized Chebyshev neighborhood utilities using fast memory management 
#     and implicit BLAS/LAPACK operations.
# """

# using LinearAlgebra
# using NearestNeighbors

# function chev_nbd_knn(domain_grid::AbstractMatrix{T},
#                       cheb_grid::AbstractMatrix{T},
#                       l::Int
# ) where {T<:Real}

#     # 1. Build tree once
#     tree = KDTree(permutedims(domain_grid))

#     pp = size(cheb_grid, 1)
#     dim = size(domain_grid, 2)
    
#     # Pre-allocate indices collector to entirely bypass `vcat` allocations
#     keep_idxs = Vector{Int}(undef, pp * l)
#     count_collected = 0

#     selected_mask = falses(size(domain_grid, 1))
#     k_extra = min(5l + 10, size(domain_grid, 1))

#     for i in 1:pp
#         # Use an allocation-free view for the center point
#         center = @view cheb_grid[i, :]

#         idxs, _ = knn(tree, center, k_extra, true)

#         added_for_this_node = 0
#         for idx in idxs
#             if !selected_mask[idx]
#                 selected_mask[idx] = true
#                 count_collected += 1
#                 keep_idxs[count_collected] = idx
#                 added_for_this_node += 1
                
#                 if added_for_this_node == l
#                     break
#                 end
#             end
#         end
#     end

#     # Single allocation at the very end 
#     return domain_grid[view(keep_idxs, 1:count_collected), :]
# end


# # ---------------------------------------------------------
# # Low-rank approximation from CUR (tolerance version)
# # ---------------------------------------------------------
# function cheb_nbd_lowrank_from_construction_tol(
#     target_grid::AbstractMatrix{<:Real},
#     source_grid::AbstractMatrix{<:Real},
#     x_cheb_grid::AbstractMatrix{<:Real},
#     y_cheb_grid::AbstractMatrix{<:Real},
#     l::Int,
#     tol::Real,
#     kernel_type::String
# )

#     # --------------------------------
#     # 1. Construct neighborhoods (No vcat bottlenecks)
#     # --------------------------------
#     X_CHEB_NBDs = chev_nbd_knn(target_grid, x_cheb_grid, l)
#     Y_CHEB_NBDs = chev_nbd_knn(source_grid, y_cheb_grid, l)

#     # --------------------------------
#     # 2. Construct CUR matrices
#     # --------------------------------
#     C = get_matrix_from_grid_nodes(target_grid, Y_CHEB_NBDs, kernel_type)
#     U = get_matrix_from_grid_nodes(X_CHEB_NBDs, Y_CHEB_NBDs, kernel_type)
#     R = get_matrix_from_grid_nodes(X_CHEB_NBDs, source_grid, kernel_type)

#     # --------------------------------
#     # 3. Economy QR using IMPLICIT BLAS
#     # --------------------------------
#     Fc = qr(C)
#     Fr = qr(transpose(R))

#     # Keep Rc and Rr as upper triangular matrix slices (highly efficient)
#     Rc = Fc.R
#     Rr = Fr.R

#     # --------------------------------
#     # 4. Core matrix calculation (BLAS multi-threaded backslash)
#     # --------------------------------
#     X = U \ transpose(Rr)
#     S = Rc * X

#     # --------------------------------
#     # 5. SVD of small core (Highly optimized LAPACK call)
#     # --------------------------------
#     Us, singular_vals, Vs = svd(S)

#     max_sv = singular_vals[1]
#     r_eff = count(sv -> sv >= tol * max_sv, singular_vals)

#     # Allocate views to avoid copying the singular vectors
#     Ur_small = @view Us[:, 1:r_eff]
#     Sr       = singular_vals[1:r_eff]
#     Vr_small = @view Vs[:, 1:r_eff]

#     # --------------------------------
#     # 6. Lift back using implicit multi-threaded BLAS/LAPACK multiplication
#     # --------------------------------
#     # Fc.Q and Fr.Q apply Householder reflections directly via BLAS 3 routines, 
#     # skipping explicit N x N matrix construction entirely.
#     Utilde = Fc.Q * Ur_small
#     Vtilde = Fr.Q * Vr_small

#     return Utilde, Diagonal(Sr), Vtilde, r_eff
# end


# # ---------------------------------------------------------
# # Low-rank approximation from CUR (rank version)
# # ---------------------------------------------------------
# function cheb_nbd_lowrank_from_construction(
#     target_grid::AbstractMatrix{<:Real},
#     source_grid::AbstractMatrix{<:Real},
#     x_cheb_grid::AbstractMatrix{<:Real},
#     y_cheb_grid::AbstractMatrix{<:Real},
#     l::Int,
#     r::Int,
#     kernel_type::String
# )
#     X_CHEB_NBDs = chev_nbd_knn(target_grid, x_cheb_grid, l)
#     Y_CHEB_NBDs = chev_nbd_knn(source_grid, y_cheb_grid, l)

#     C = get_matrix_from_grid_nodes(target_grid, Y_CHEB_NBDs, kernel_type)
#     U = get_matrix_from_grid_nodes(X_CHEB_NBDs, Y_CHEB_NBDs, kernel_type)
#     R = get_matrix_from_grid_nodes(X_CHEB_NBDs, source_grid, kernel_type)

#     Fc = qr(C)
#     Fr = qr(transpose(R))

#     Rc = Fc.R
#     Rr = Fr.R

#     X = U \ transpose(Rr)
#     S = Rc * X

#     Us, Ss, Vs = svd(S)

#     Ur_small = @view Us[:, 1:r]
#     Sr       = Ss[1:r]
#     Vr_small = @view Vs[:, 1:r]

#     Utilde = Fc.Q * Ur_small
#     Vtilde = Fr.Q * Vr_small

#     return Utilde, Diagonal(Sr), Vtilde
# end