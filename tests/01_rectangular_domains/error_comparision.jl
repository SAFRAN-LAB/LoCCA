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
# julia -t 8 tests/01_rectangular_domains/error_comparision.jl | tee OUTPUTS/log_K4_tol_1.0e-9.txt
# rememver that you have to be in cheb_nbd_nodes directory to run the above command
# =============================================================================================


# Load required files
include(joinpath(@__DIR__, "..", "..", "src", "kernels.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "cheb_nbd_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "LoCCA_SI_utils.jl"))
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


    println("\n============================================================")
    println("          CONDITIONING OF LoCCA SKELETON")
    println("============================================================")

    println("Size of U_LoCCA = ", size(U_LoCCA))

    # # Condition number
    # condU = cond(U_LoCCA)

    # # Norm of inverse
    # invU = inv(U_LoCCA)
    # normInv2 = opnorm(invU)
    # normInvInf = opnorm(invU, Inf)

    # # Norm of skeleton matrix
    # normU2 = opnorm(U_LoCCA)
    # normUInf = opnorm(U_LoCCA, Inf)

    # println("cond(U_LoCCA)              = ", @sprintf("%.4e", condU))
    # println("||U_LoCCA||₂              = ", @sprintf("%.4e", normU2))
    # println("||U_LoCCA||∞              = ", @sprintf("%.4e", normUInf))
    # println("||U_LoCCA^{-1}||₂         = ", @sprintf("%.4e", normInv2))
    # println("||U_LoCCA^{-1}||∞         = ", @sprintf("%.4e", normInvInf))

    # BasisLeft = C_LoCCA * inv(U_LoCCA)
    # BasisRight = inv(U_LoCCA) * R_LoCCA

    # println("||C*U^{-1}||₂   = ", opnorm(BasisLeft))
    # println("||C*U^{-1}||∞   = ", opnorm(BasisLeft, Inf))

    # println("||U^{-1}*R||₂   = ", opnorm(BasisRight))
    # println("||U^{-1}*R||∞   = ", opnorm(BasisRight, Inf))

    # println("============================================================")


    

    K_diff = copy(K_actual)
    axpy!(-1.0, K_approx, K_diff)

    linspace_cheb_error = norm(K_diff) / norm_K_actual


    # ---------------------------------------------------------
    #                   SVD (same rank r)
    # ---------------------------------------------------------
    U_svd, S_svd, V_svd = svd(K_actual)

    K_svd_approx = U_svd[:, 1:r] * Diagonal(S_svd[1:r]) * V_svd[:, 1:r]'

    K_diff = copy(K_actual)
    axpy!(-1.0, K_svd_approx, K_diff)

    linspace_svd_error = norm(K_diff) / norm_K_actual

    # ---------------------------------------------------------
    #                    ACA (same rank r)
    # ---------------------------------------------------------
    I, J, _, _, _ = perform_ppACA_rank(kernel_choice, target_grid, source_grid, r, 1)

    C_ppACA = get_matrix_from_grid_nodes(target_grid, source_grid[J, :], kernel_choice)

    U_ppACA = get_matrix_from_grid_nodes(target_grid[I, :], source_grid[J, :], kernel_choice)

    R_ppACA = get_matrix_from_grid_nodes(target_grid[I, :], source_grid, kernel_choice)

    R_hat_ppACA = U_ppACA \ R_ppACA
    K_approx_ppACA = C_ppACA * R_hat_ppACA

    K_diff = copy(K_actual)
    axpy!(-1.0, K_approx_ppACA, K_diff)

    linspace_ppACA_error = norm(K_diff) / norm_K_actual

    # ---------------------------------------------------------
    #                      SI (same rank r)
    # ---------------------------------------------------------
    Xhat, Yhat, _, _, _ = skeletonized_interpolation_nodes_rank(kernel_choice, target_hypercube,
                                source_hypercube, 7, r)                # change 11 to some other number if better accuracy is desired for SI node selection

    C_si = get_matrix_from_grid_nodes(target_grid, Yhat, kernel_choice)

    U_si = get_matrix_from_grid_nodes(Xhat, Yhat, kernel_choice)

    R_si = get_matrix_from_grid_nodes(Xhat, source_grid, kernel_choice)

    R_hat_si = U_si \ R_si
    K_approx_si = C_si * R_hat_si

    K_diff = copy(K_actual)
    axpy!(-1.0, K_approx_si, K_diff)

    linspace_si_error = norm(K_diff) / norm_K_actual

    # ---------------------------------------------------------
    #                   Print linspace errors
    # ---------------------------------------------------------
    @printf("LoCCA Error : %.6e\n", linspace_cheb_error)
    println("Obtained rank is r = ", r, " which was used for the following approximations:")
    println("------------------------------------------------------------")
    @printf("SVD Error      : %.6e\n", linspace_svd_error)
    @printf("ACA Error      : %.6e\n", linspace_ppACA_error)
    @printf("SI Error       : %.6e\n", linspace_si_error)
    println("------------------------------------------------------------")

    # breakpoint()  # Set a breakpoint here to inspect the errors and rank before moving on to the random node experiments

    # ==========================================================
    #               RANDOM NODE EXPERIMENTS
    # ==========================================================

    node_type = "random"

    println("\n============================================================")
    println("               RANDOM GRID EXPERIMENTS")
    println("============================================================")

    cheb_nbd_error = zeros(N_test)
    svd_error      = zeros(N_test)
    ppACA_error    = zeros(N_test)
    si_error       = zeros(N_test)

    Threads.@threads for test in 1:N_test
        # for test in 1:N_test

        # 1. Force all loop-scoped variables to be local to prevent data races
        local XX_axes, YY_axes, target_grid, source_grid
        local K_actual, norm_K_actual, K_diff, temp
        local Utilde, Sr, Vtilde, K_approx
        local U_svd, S_svd, V_svd, K_svd_approx
        local I, J, C_ppACA, U_ppACA, R_ppACA, R_hat_ppACA, K_approx_ppACA
        local C_si, U_si, R_si, R_hat_si, K_approx_si

        # rng = MersenneTwister(1000 + test)

        XX_axes, YY_axes = generate_hypercube_axes(source_hypercube, target_hypercube, NN, node_type)

        target_grid = generate_grid_points(XX_axes)
        source_grid = generate_grid_points(YY_axes)

        K_actual = get_matrix_from_grid_nodes(target_grid, source_grid, kernel_choice)

        norm_K_actual = norm(K_actual)

        # -----------------------------------------------------
        #                LoCCA (fixed rank r)
        # -----------------------------------------------------
        I_LoCCA, J_LoCCA, _ =
                    LoCCA_with_grid_input_rank(
                        target_grid,
                        source_grid,
                        x_cheb_grid,
                        y_cheb_grid,
                        l,
                        kernel_choice,
                        r
                    )
        C_LoCCA = get_matrix_from_grid_nodes(target_grid, J_LoCCA, kernel_choice)
        U_LoCCA = get_matrix_from_grid_nodes(I_LoCCA, J_LoCCA, kernel_choice)
        R_LoCCA = get_matrix_from_grid_nodes(I_LoCCA, source_grid, kernel_choice)

        R_hat_LoCCA = U_LoCCA \ R_LoCCA
        K_approx = C_LoCCA * R_hat_LoCCA

        K_diff = copy(K_actual)
        axpy!(-1.0, K_approx, K_diff)

        cheb_nbd_error[test] = norm(K_diff) / norm_K_actual

        # -----------------------------------------------------
        #                       SVD
        # -----------------------------------------------------
        U_svd, S_svd, V_svd = svd(K_actual)

        K_svd_approx = U_svd[:,1:r] * Diagonal(S_svd[1:r]) * V_svd[:,1:r]'

        K_diff = copy(K_actual)
        axpy!(-1.0, K_svd_approx, K_diff)

        svd_error[test] = norm(K_diff) / norm_K_actual

        # -----------------------------------------------------
        #                      ppACA
        # -----------------------------------------------------
        I, J, _, _, _ = perform_ppACA_rank(kernel_choice, target_grid, source_grid, r, 1)

        C_ppACA = get_matrix_from_grid_nodes(target_grid, source_grid[J,:], kernel_choice)

        U_ppACA = get_matrix_from_grid_nodes(target_grid[I,:], source_grid[J,:], kernel_choice)

        R_ppACA = get_matrix_from_grid_nodes(target_grid[I,:], source_grid, kernel_choice)

        R_hat_ppACA = U_ppACA \ R_ppACA
        K_approx_ppACA = C_ppACA * R_hat_ppACA

        K_diff = copy(K_actual)
        axpy!(-1.0, K_approx_ppACA, K_diff)

        ppACA_error[test] = norm(K_diff) / norm_K_actual

        # -----------------------------------------------------
        #                       SI
        # -----------------------------------------------------
        C_si = get_matrix_from_grid_nodes(target_grid, Yhat, kernel_choice)
            
        U_si = get_matrix_from_grid_nodes(Xhat, Yhat, kernel_choice)

        R_si = get_matrix_from_grid_nodes( Xhat, source_grid, kernel_choice)

        R_hat_si = U_si \ R_si
        K_approx_si = C_si * R_hat_si

        K_diff = copy(K_actual)
        axpy!(-1.0, K_approx_si, K_diff)

        si_error[test] = norm(K_diff) / norm_K_actual
    end

    # ==========================================================
    #                       Summary Statistics
    # ==========================================================
    println("\n============================================================")
    println("               SUMMARY STATISTICS")
    println("------------------------------------------------------------")

    @printf("Mean LoCCA Error    : %.6e\n", mean(cheb_nbd_error))

    @printf("Mean SVD Error      : %.6e\n", mean(svd_error))

    @printf("Mean ACA Error      : %.6e\n", mean(ppACA_error))

    @printf("Mean SI Error       : %.6e\n", mean(si_error))

    println("------------------------------------------------------------")
    println("NOTE: The same rank (r = ", r, ") was used for all approximations done in the random grid case.")
    println("============================================================")

    # # ==========================================================
    # #                        Save CSV
    # # ==========================================================
    # df = DataFrame(LoCCA = cheb_nbd_error, SVD = svd_error,
    #                 ACA = ppACA_error, SI = si_error)

    # output_file = joinpath(@__DIR__, "..", "..", "OUTPUTS",
    #         "error_summary_$(kernel_choice)_tol_$(tol).csv")

    # CSV.write(output_file, df)

    # println("\nSaved error data to:")
    # println(output_file)

    # println("\n============================================================\n")

end

main()