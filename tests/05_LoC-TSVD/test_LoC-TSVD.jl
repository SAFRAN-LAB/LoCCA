# =============================================================================================
# LoC-TSVD Validation Extension
# Author : Sumit Singh, Sivaram Ambikasaran and collaborators
# Date   : May 2026
# Description: This script benchmarks the low-rank approximations obtained by LoC-TSVD
# on a uniform grid geometry, validates spectral subspace congruence against the exact SVD,
# and saves the matching singular values to a dynamically structured CSV file.
# =============================================================================================

print("\033[2J\033[H")
flush(stdout)

# Load required files
include(joinpath(@__DIR__, "..", "..", "src", "kernels.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "cheb_nbd_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "geometry_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "kernel_matrix_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "ppACA.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "chebyshev_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "SI_utils.jl"))

# Load required packages
using Printf
using LinearAlgebra
BLAS.set_num_threads(2)  
using Base.Threads
using Statistics
using CSV
using DataFrames
using Random

# Load config
include(joinpath(@__DIR__, "config.jl"))
using .Test_Config

function main()

    pms = PARAMS

    kernel_choice = pms.kernel_choice
    NN       = pms.num_domain_nodes
    N_test  = pms.num_test
    p       = pms.chev_nodes_p
    l       = pms.nbd_size
    tol     = pms.tol

    target_hypercube = pms.target_hypercube
    source_hypercube = pms.source_hypercube

    dim = size(source_hypercube, 1)

    x_cheb_grid = generate_chebyshev_grid(target_hypercube, p)
    y_cheb_grid = generate_chebyshev_grid(source_hypercube, p)

    println("\n============================================================")
    println("                    EXPERIMENT DETAILS")
    println("============================================================")
    println("Kernel Choice         : ", kernel_choice)
    println("Dimension             : ", dim)
    println("Grid Size             : ", NN, "^", dim)
    println("Number of Test        : ", N_test)
    println("Chebyshev Grid Size   : ", p, "^", dim)
    println("Nbd Size l            : ", l)
    println("Target accuracy       : ", tol)
    println("Target Hypercube      : ", target_hypercube)
    println("Source Hypercube      : ", source_hypercube)
    println("============================================================\n")

    # ==========================================================
    #               UNIFORM GRID CASE (LINSPACE)
    # ==========================================================

    node_type = "linspace"

    println("\n============================================================")
    println("                 LINSPACE GRID EXPERIMENT")
    println("============================================================")

    XX_axes, YY_axes = generate_hypercube_axes(source_hypercube, target_hypercube, NN, node_type)

    target_grid = generate_grid_points(XX_axes)
    source_grid = generate_grid_points(YY_axes)

    K_actual = get_matrix_from_grid_nodes(target_grid, source_grid, kernel_choice)
    norm_K_actual = norm(K_actual)

    # ---------------------------------------------------------
    #          LoC-TSVD (obtain rank r)
    # ---------------------------------------------------------
    Utilde, Sr, Vtilde, r = cheb_nbd_lowrank_from_construction_tol(
            target_grid, source_grid, x_cheb_grid, y_cheb_grid,
            l, tol, kernel_choice)

    # Reconstruction Error Check
    temp = Utilde * Sr
    K_approx = temp * transpose(Vtilde)
    K_diff = copy(K_actual)
    axpy!(-1.0, K_approx, K_diff)
    linspace_loc_error = norm(K_diff) / norm_K_actual

    # ---------------------------------------------------------
    #                   True Analytical SVD 
    # ---------------------------------------------------------
    U_svd, S_svd, V_svd = svd(K_actual)

    K_svd_approx = U_svd[:, 1:r] * Diagonal(S_svd[1:r]) * V_svd[:, 1:r]'
    K_diff_svd = copy(K_actual)
    axpy!(-1.0, K_svd_approx, K_diff_svd)
    linspace_svd_error = norm(K_diff_svd) / norm_K_actual

    # ---------------------------------------------------------
    #        Subspace Consistency and Orthogonality Drift Metrics
    # ---------------------------------------------------------
    U_svd_truncated = U_svd[:, 1:r]

    Cross_Product = Utilde' * U_svd_truncated
    S_cross = svdvals(Cross_Product)

    cos_theta_max = minimum(S_cross)
    E_sub = sqrt(max(0.0, 1.0 - cos_theta_max^2))

    E_orth = maximum(abs, Utilde' * Utilde - I)
    E_orth_cross = maximum(abs, abs.(Cross_Product) - I)

    # ---------------------------------------------------------
    #        Extract and Save Singular Values to CSV
    # ---------------------------------------------------------
    # Extract the raw singular values vector from the Diagonal matrix wrapper
    loc_svd_vals = diag(Sr)
    actual_svd_vals = S_svd[1:r]

    # Create the DataFrame with the requested two columns
    df_sv = DataFrame(
        LoC_TSVD_Singular_Values = loc_svd_vals,
        Actual_SVD_Singular_Values = actual_svd_vals
    )

    # Dynamic filename configuration based on parameters
    filename = "singular_values_$(kernel_choice)_tol_$(tol).csv"
    
    # Save file (it writes directly inside the current working directory)
    CSV.write(filename, df_sv)
    println("Saved matching singular values to: ", filename)

    # ---------------------------------------------------------
    #                   Print Unified Validation Data
    # ---------------------------------------------------------
    println("\n------------------------------------------------------------")
    println("Obtained rank of the approximated matrix is r = ", r)
    println("------------------------------------------------------------")
    @printf("LoC-TSVD Relative Error  : %.6e\n", linspace_loc_error)
    @printf("True SVD Truncated Error : %.6e\n", linspace_svd_error)
    println("------------------------------------------------------------")
    @printf("Subspace Error E_sub (2-norm)     : %.6e\n", E_sub)
    @printf("Orthogonality Drift E_orth (max)  : %.6e\n", E_orth)
    println("------------------------------------------------------------")

end

main()