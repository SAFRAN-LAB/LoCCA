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

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
Pkg.instantiate()

# =============================================================================================
# To save the terminl outputs to a file, run the script with:
# julia test_run/test_LoCCA.jl
# rememver that you have to be in cheb_nbd_nodes directory to run the above command
# =============================================================================================


# Load required files
include(joinpath(@__DIR__, "..", "src", "kernels.jl"))
include(joinpath(@__DIR__, "..", "src", "cheb_nbd_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "LoCCA_SI_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "geometry_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "kernel_matrix_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "chebyshev_utils.jl"))

# Load required packages
using Printf
using LinearAlgebra

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


    XX_axes, YY_axes = generate_hypercube_axes(source_hypercube, target_hypercube, NN, node_type)

    target_grid = generate_grid_points(XX_axes)
    source_grid = generate_grid_points(YY_axes)

    K_actual = get_matrix_from_grid_nodes( target_grid, source_grid, kernel_choice)

    norm_K_actual = norm(K_actual)

    # ---------------------------------------------------------
    #          LoCCA with tolerance (obtain rank r)
    # ---------------------------------------------------------
    I_LoCCA, J_LoCCA, r =
                LoCCA_with_grid_input(
                    target_grid,
                    source_grid,
                    x_cheb_grid,
                    y_cheb_grid,
                    l,
                    kernel_choice,
                    tol
                )
    C_LoCCA = get_matrix_from_grid_nodes(target_grid, J_LoCCA, kernel_choice)
    U_LoCCA = get_matrix_from_grid_nodes(I_LoCCA, J_LoCCA, kernel_choice)
    R_LoCCA = get_matrix_from_grid_nodes(I_LoCCA, source_grid, kernel_choice)

    R_hat_LoCCA = U_LoCCA \ R_LoCCA
    K_approx = C_LoCCA * R_hat_LoCCA    

    K_diff = copy(K_actual)
    axpy!(-1.0, K_approx, K_diff)

    LoCCA_error = norm(K_diff) / norm_K_actual

    # ---------------------------------------------------------
    #                   Print linspace errors
    # ---------------------------------------------------------
    @printf("LoCCA Error : %.6e\n", LoCCA_error)
    println("Obtained rank is r = ", r)
    println("------------------------------------------------------------")


    # ---------------------------------------------------------
    # Verification confirmation for external users
    # ---------------------------------------------------------
    if LoCCA_error <= tol * 10.0
        println(" [PASS] Quick test completed successfully.")
        println("        • Core modules and dependencies are verified.")
        println("        • You can now run the other experiment scripts.\n")
    else
        println(" [WARN] LoCCA ran to completion, but relative error was above expected tolerance.")
        println("        Check your parameters in config.jl before starting full runs.\n")
    end


end

main()