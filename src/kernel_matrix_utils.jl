# ============================================================
# kernel_matrix_utils.jl
# Kernel Matrix Utilities (Axis-based + Node-based)
# ============================================================

using LinearAlgebra
using Distances

# ------------------------------------------------------------
# Generate Tensor Product Grid from Axes
# Input:  coords (Dim × N_axis)
# Output: (N_axis^Dim × Dim)
# ------------------------------------------------------------
function generate_grid_points(coords::Matrix{<:Real})

    Dim, n_axis = size(coords)

    if Dim == 1
        return coords'
    end

    # Tensor product using Iterators.product
    grids = Iterators.product((coords[d, :] for d in 1:Dim)...)

    N_total = n_axis^Dim
    grid_points = Matrix{Float64}(undef, N_total, Dim)

    i = 1
    for g in grids
        grid_points[i, :] .= g
        i += 1
    end

    return grid_points
end


# ------------------------------------------------------------
# Compute Kernel Matrix from Axis Grids
# target, source: (Dim × N_axis)
# Returns: (N_target^Dim × N_source^Dim)
# ------------------------------------------------------------
function get_kernel_matrix(target::Matrix{<:Real},
                           source::Matrix{<:Real},
                           kernel_choice::String)

    Dim, n_target = size(target)
    _,   n_source = size(source)

    # Memory warning
    total_target = n_target^Dim
    total_source = n_source^Dim

    if total_target > 1e6 || total_source > 1e6
        @warn "Large grid detected." target_points=total_target source_points=total_source
    end

    # Generate full tensor grids
    X_grid = generate_grid_points(target)
    Y_grid = generate_grid_points(source)

    # Compute pairwise Euclidean distances
    # Distances.jl expects (Dim × N)
    D = pairwise(Euclidean(), X_grid', Y_grid')

    # Apply kernel
    return ker_fun(D, kernel_choice)
end


# ------------------------------------------------------------
# Compute Kernel Matrix from Full Grid Nodes
# target: (N_X × Dim)
# source: (N_Y × Dim)
# ------------------------------------------------------------
function get_matrix_from_grid_nodes(target::AbstractMatrix{<:Real},
                                    source::AbstractMatrix{<:Real},
                                    kernel_choice::String)

    # Compute distances directly
    D = pairwise(Euclidean(), target', source')

    return ker_fun(D, kernel_choice)
end
