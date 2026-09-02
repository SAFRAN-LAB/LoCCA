# =============================================================================================
# LoCCA  : Localized Chebyshev Cross Approximation
# Author : Sumit Singh, Sivaram Ambikasaran and collaborators
# Date   : May 2026
# Description: This script tests the accuracy of the low-rank approximations obtained by LoCCA
# on both uniform and random target and source grids using random sub-sampling for large N. 
# The errors are compared against ACA and SI approximations of the same rank. 
# The error data from the random grid experiments is saved to a CSV file.
# =============================================================================================

print("\033[2J\033[H")
flush(stdout)


# =============================================================================================
# To save the terminl outputs to a file, run the script with:
# julia -t 8 zz_LoCCA_SI/error_comparision_large_N.jl | tee OUTPUTS/log_K4_tol_1.0e-9.txt
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
BLAS.set_num_threads(2)  # Adjust as needed
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
    NN            = pms.num_domain_nodes # Grid size per dimension
    N_test        = pms.num_test
    p             = pms.chev_nodes_p
    l             = pms.nbd_size
    tol           = pms.tol

    # ---------------------------------------------------------
    # Hypercube definitions
    # ---------------------------------------------------------
    target_hypercube = pms.target_hypercube
    source_hypercube = pms.source_hypercube

    dim = size(source_hypercube, 1)
    total_nodes = NN^dim
    
    # Maximum subset size for error estimation
    n_sample = min(2000, total_nodes)  

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
    println("Grid Size             : ", NN, "^", dim, " (Total Nodes: ", total_nodes, ")")
    println("Number of Test        : ", N_test)
    println("Chebyshev Grid Size   : ", p, "^", dim)
    println("Nbd Size l            : ", l)
    println("Target accuracy       : ", tol)
    println("Sample Size for Error : ", n_sample)
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

    # ---------------------------------------------------------
    # Sub-sampling Setup for Linspace Error Evaluation
    # ---------------------------------------------------------
    Random.seed!(42) 
    sampled_rows = randperm(total_nodes)[1:n_sample]
    sampled_cols = randperm(total_nodes)[1:n_sample]

    sub_target_grid = target_grid[sampled_rows, :]
    sub_source_grid = source_grid[sampled_cols, :]

    K_true_sub = get_matrix_from_grid_nodes(sub_target_grid, sub_source_grid, kernel_choice)
    norm_K_true = norm(K_true_sub)

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
    
    C_LoCCA_sub = get_matrix_from_grid_nodes(sub_target_grid, J_LoCCA, kernel_choice)
    U_LoCCA_core = get_matrix_from_grid_nodes(I_LoCCA, J_LoCCA, kernel_choice)
    R_LoCCA_sub = get_matrix_from_grid_nodes(I_LoCCA, sub_source_grid, kernel_choice)

    R_hat_LoCCA_sub = U_LoCCA_core \ R_LoCCA_sub
    K_approx_LoCCA_sub = C_LoCCA_sub * R_hat_LoCCA_sub
    
    # High performance error calculation using axpy!
    K_diff = copy(K_true_sub)
    axpy!(-1.0, K_approx_LoCCA_sub, K_diff)
    linspace_cheb_error = norm(K_diff) / norm_K_true

    

    # ---------------------------------------------------------
    #                    ACA (same rank r)
    # ---------------------------------------------------------
    I, J, _, _ =
                perform_ppACA_rank_fast(
                    kernel_choice,
                    target_grid,
                    source_grid,
                    r,
                    1
                )

    C_ppACA_sub = get_matrix_from_grid_nodes(sub_target_grid, source_grid[J, :], kernel_choice)
    U_ppACA     = get_matrix_from_grid_nodes(target_grid[I, :], source_grid[J, :], kernel_choice)
    R_ppACA_sub = get_matrix_from_grid_nodes(target_grid[I, :], sub_source_grid, kernel_choice)

    

    R_hat_ppACA_sub = U_ppACA \ R_ppACA_sub
    K_approx_ppACA_sub = C_ppACA_sub * R_hat_ppACA_sub

    # High performance error calculation using axpy!
    K_diff = copy(K_true_sub)
    axpy!(-1.0, K_approx_ppACA_sub, K_diff)
    linspace_ppACA_error = norm(K_diff) / norm_K_true

    

    # ---------------------------------------------------------
    #                      SI (same rank r)
    # ---------------------------------------------------------
    Xhat, Yhat, _, _, _ = skeletonized_interpolation_nodes_rank(kernel_choice, target_hypercube,
                                source_hypercube, p, r) 

    C_si_sub = get_matrix_from_grid_nodes(sub_target_grid, Yhat, kernel_choice)
    U_si     = get_matrix_from_grid_nodes(Xhat, Yhat, kernel_choice)
    R_si_sub = get_matrix_from_grid_nodes(Xhat, sub_source_grid, kernel_choice)

    R_hat_si_sub = U_si \ R_si_sub
    K_approx_si_sub = C_si_sub * R_hat_si_sub

    # High performance error calculation using axpy!
    K_diff = copy(K_true_sub)
    axpy!(-1.0, K_approx_si_sub, K_diff)
    linspace_si_error = norm(K_diff) / norm_K_true
    

    # ---------------------------------------------------------
    #                   Print linspace errors
    # ---------------------------------------------------------
    @printf("LoCCA Error : %.6e\n", linspace_cheb_error)
    println("Obtained rank is r = ", r, " which was used for the following approximations:")
    println("------------------------------------------------------------")
    @printf("ACA Error      : %.6e\n", linspace_ppACA_error)
    @printf("SI Error       : %.6e\n", linspace_si_error)
    println("------------------------------------------------------------")


    # ==========================================================
    #               RANDOM NODE EXPERIMENTS
    # ==========================================================

    node_type = "random"

    println("\n============================================================")
    println("               RANDOM GRID EXPERIMENTS")
    println("============================================================")

    cheb_nbd_error = zeros(N_test)
    ppACA_error    = zeros(N_test)
    si_error       = zeros(N_test)

    Threads.@threads for test in 1:N_test

        # 1. Force all loop-scoped variables to be local to prevent data races
        local XX_axes, YY_axes, target_grid, source_grid
        local K_true_sub, norm_K_true
        local I_LoCCA, J_LoCCA, C_LoCCA_sub, U_LoCCA_core, R_LoCCA_sub, R_hat_LoCCA_sub, K_approx_LoCCA_sub
        local I, J, C_ppACA_sub, U_ppACA, R_ppACA_sub, R_hat_ppACA_sub, K_approx_ppACA_sub
        local C_si_sub, U_si, R_si_sub, R_hat_si_sub, K_approx_si_sub
        local sampled_rows, sampled_cols, sub_target_grid, sub_source_grid

        # Thread-local RNG initialization
        rng = MersenneTwister(1000 + test)

        XX_axes, YY_axes = generate_hypercube_axes(source_hypercube, target_hypercube, NN, node_type)

        target_grid = generate_grid_points(XX_axes)
        source_grid = generate_grid_points(YY_axes)

        # Thread-safe random sub-sampling indices using the local rng
        sampled_rows = randperm(rng, total_nodes)[1:n_sample]
        sampled_cols = randperm(rng, total_nodes)[1:n_sample]

        sub_target_grid = target_grid[sampled_rows, :]
        sub_source_grid = source_grid[sampled_cols, :]

        K_true_sub = get_matrix_from_grid_nodes(sub_target_grid, sub_source_grid, kernel_choice)
        norm_K_true = norm(K_true_sub)

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
        
        C_LoCCA_sub = get_matrix_from_grid_nodes(sub_target_grid, J_LoCCA, kernel_choice)
        U_LoCCA_core = get_matrix_from_grid_nodes(I_LoCCA, J_LoCCA, kernel_choice)
        R_LoCCA_sub = get_matrix_from_grid_nodes(I_LoCCA, sub_source_grid, kernel_choice)

        R_hat_LoCCA_sub = U_LoCCA_core \ R_LoCCA_sub
        K_approx_LoCCA_sub = C_LoCCA_sub * R_hat_LoCCA_sub

        cheb_nbd_error[test] = norm(K_true_sub - K_approx_LoCCA_sub) / norm_K_true

        # -----------------------------------------------------
        #                      ppACA
        # -----------------------------------------------------
        I, J, _, _ =
                perform_ppACA_rank_fast(
                    kernel_choice,
                    target_grid,
                    source_grid,
                    r,
                    1
                )

        C_ppACA_sub = get_matrix_from_grid_nodes(sub_target_grid, source_grid[J, :], kernel_choice)
        U_ppACA     = get_matrix_from_grid_nodes(target_grid[I, :], source_grid[J, :], kernel_choice)
        R_ppACA_sub = get_matrix_from_grid_nodes(target_grid[I, :], sub_source_grid, kernel_choice)

        R_hat_ppACA_sub = U_ppACA \ R_ppACA_sub
        K_approx_ppACA_sub = C_ppACA_sub * R_hat_ppACA_sub

        ppACA_error[test] = norm(K_true_sub - K_approx_ppACA_sub) / norm_K_true

        # -----------------------------------------------------
        #                       SI
        # -----------------------------------------------------
        C_si_sub = get_matrix_from_grid_nodes(sub_target_grid, Yhat, kernel_choice)
        U_si     = get_matrix_from_grid_nodes(Xhat, Yhat, kernel_choice)
        R_si_sub = get_matrix_from_grid_nodes(Xhat, sub_source_grid, kernel_choice)

        R_hat_si_sub = U_si \ R_si_sub
        K_approx_si_sub = C_si_sub * R_hat_si_sub

        si_error[test] = norm(K_true_sub - K_approx_si_sub) / norm_K_true
    end

    # ==========================================================
    #                       Summary Statistics
    # ==========================================================
    println("\n============================================================")
    println("               SUMMARY STATISTICS")
    println("------------------------------------------------------------")

    @printf("Mean LoCCA Error    : %.6e\n", mean(cheb_nbd_error))
    @printf("Mean ACA Error      : %.6e\n", mean(ppACA_error))
    @printf("Mean SI Error       : %.6e\n", mean(si_error))

    println("------------------------------------------------------------")
    println("NOTE: The same rank (r = ", r, ") was used for all approximations done in the random grid case.")
    println("============================================================")

    # ==========================================================
    #                        Save CSV
    # ==========================================================
    df = DataFrame(LoCCA = cheb_nbd_error, ACA = ppACA_error, SI = si_error)

    output_file = joinpath(@__DIR__, "..", "..", "OUTPUTS",
            "error_summary_$(kernel_choice)_tol_$(tol).csv")

    CSV.write(output_file, df)

    println("\nSaved error data to:")
    println(output_file)

    println("\n============================================================\n")

end

main()