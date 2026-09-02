# =============================================================================================
# LoCCA  : Localized Chebyshev Cross Approximation
# Author : Sumit Singh, Sivaram Ambikasaran and collaborators
# Date   : May 2026
# Description: This script tests the accuracy of the low-rank approximations obtained by LoCCA
# on both uniform and random target and source grids. The errors are compared against SVD, ACA,
# and SI approximations of the same rank. The error data from the random grid experiments is 
# saved to a CSV file for further analysis.
# =============================================================================================

print("\033[2J\033[H")
flush(stdout)

# =============================================================================================
# To save the terminl outputs to a file, run the script with:
# julia -t 8 tests/error_testing.jl | tee OUTPUTS/log_K4_tol_1.0e-9.txt
# rememver that you have to be in cheb_nbd_nodes directory to run the above command
# =============================================================================================


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
BLAS.set_num_threads(2)  # Set the number of threads for BLAS operations to 1 (adjust as needed)
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

    # ---------------------------------------------------------
    # Hypercube definitions
    # ---------------------------------------------------------
    target_hypercube = pms.target_hypercube
    source_hypercube = pms.source_hypercube

    dim = size(source_hypercube, 1)

    # ---------------------------------------------------------
    # Generate Chebyshev grids
    # ---------------------------------------------------------
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
    #               LINSPACE CASE (RUN ONLY ONCE)
    # ==========================================================

    node_type = "linspace"

    println("\n============================================================")
    println("                 LINSPACE GRID EXPERIMENT")
    println("============================================================")


    XX_axes, YY_axes = generate_hypercube_axes(source_hypercube, target_hypercube, NN, node_type)

    target_grid = generate_grid_points(XX_axes)
    source_grid = generate_grid_points(YY_axes)

    K_actual = get_matrix_from_grid_nodes( target_grid, source_grid, kernel_choice)

    norm_K_actual = norm(K_actual)

    # ---------------------------------------------------------
    #          LoCCA with tolerance (obtain rank r)
    # ---------------------------------------------------------
    Utilde, Sr, Vtilde, r = cheb_nbd_lowrank_from_construction_tol(
            target_grid, source_grid, x_cheb_grid, y_cheb_grid,
            l, tol, kernel_choice)

    
    # Check if U' * U is approximately the Identity matrix
    is_orthonormal = Utilde' * Utilde ≈ I

    println("Are columns of Utilde orthonormal? ", is_orthonormal)

    is_orthonormal = Vtilde' * Vtilde ≈ I
    println("Are columns of Vtilde orthonormal? ", is_orthonormal)


    println("\nfirst and r^th singular values obtained from LoCCA: ", Sr[1], " and ", Sr[end])



    temp = Utilde * Sr
    K_approx = temp * transpose(Vtilde)

    K_diff = copy(K_actual)
    axpy!(-1.0, K_approx, K_diff)

    linspace_cheb_error = norm(K_diff) / norm_K_actual


    # ---------------------------------------------------------
    #                   SVD (same rank r)
    # ---------------------------------------------------------
    U_svd, S_svd, V_svd = svd(K_actual)

    println("first and r^th singular values obtained from SVD:   ", S_svd[1], " and ", S_svd[r])

    K_svd_approx = U_svd[:, 1:r] * Diagonal(S_svd[1:r]) * V_svd[:, 1:r]'

    K_diff = copy(K_actual)
    axpy!(-1.0, K_svd_approx, K_diff)

    linspace_svd_error = norm(K_diff) / norm_K_actual



    # ---------------------------------------------------------
    #                   Print linspace errors
    # ---------------------------------------------------------
    println("\n------------------------------------------------------------")
    println("Obtained rank of the approximated matrix is r = ", r)
    println("------------------------------------------------------------")
    @printf("LoCCA Error    : %.6e\n", linspace_cheb_error)
    @printf("SVD Error      : %.6e\n", linspace_svd_error)
    println("------------------------------------------------------------")

end

main()