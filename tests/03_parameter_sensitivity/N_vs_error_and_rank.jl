
print("\033[2J\033[H")
flush(stdout)

# =============================================================================================
# To save terminal outputs to a file, run:
#
# julia -t 8 tests/N_vs_error_and_rank.jl
#
# Remember to be inside cheb_nbd_nodes directory
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
BLAS.set_num_threads(8)

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

    # ---------------------------------------------------------
    # Fixed parameters
    # ---------------------------------------------------------
    tol = 1.0e-8
    p   = 8
    l   = 2

    # ---------------------------------------------------------
    # N values
    # ---------------------------------------------------------
    N_list = [25,35,45,55,65,75,85,95,105,115,125,135, 145]

    # ---------------------------------------------------------
    # Kernels
    # ---------------------------------------------------------
    kernel_list = [
        "kernel_1",
        "kernel_2",
        "kernel_4",
        "kernel_6",
        "kernel_7"
    ]

    # ---------------------------------------------------------
    # Hypercube definitions
    # ---------------------------------------------------------
    target_hypercube = pms.target_hypercube
    source_hypercube = pms.source_hypercube

    # ---------------------------------------------------------
    # Fixed Chebyshev grids
    # ---------------------------------------------------------
    x_cheb_grid =
        generate_chebyshev_grid(
            target_hypercube,
            p
        )

    y_cheb_grid =
        generate_chebyshev_grid(
            source_hypercube,
            p
        )

    # ==========================================================
    # Create dataframes
    # ==========================================================

    df_error = DataFrame(N = N_list)
    df_rank  = DataFrame(N = N_list)

    # ==========================================================
    # Loop over kernels
    # ==========================================================

    for kernel_choice in kernel_list

        println("\n====================================================")
        println("Running kernel: ", kernel_choice)
        println("====================================================")

        kernel_errors = Float64[]
        kernel_ranks  = Int[]

        # ==========================================================
        # Loop over N
        # ==========================================================

        for NN in N_list

            println("\n********************************************")
            println("Running N = ", NN)
            println("********************************************")

            # ---------------------------------------------------------
            # Uniform grid setup
            # ---------------------------------------------------------
            node_type = "linspace"

            XX_axes, YY_axes =
                generate_hypercube_axes(
                    source_hypercube,
                    target_hypercube,
                    NN,
                    node_type
                )

            target_grid =
                generate_grid_points(XX_axes)

            source_grid =
                generate_grid_points(YY_axes)

            # ---------------------------------------------------------
            # Actual kernel matrix
            # ---------------------------------------------------------
            K_actual =
                get_matrix_from_grid_nodes(
                    target_grid,
                    source_grid,
                    kernel_choice
                )

            norm_K_actual =
                norm(K_actual)

            # ---------------------------------------------------------
            # LoCCA approximation
            # ---------------------------------------------------------
            Utilde, Sr, Vtilde, r =
                cheb_nbd_lowrank_from_construction_tol(
                    target_grid,
                    source_grid,
                    x_cheb_grid,
                    y_cheb_grid,
                    l,
                    tol,
                    kernel_choice
                )

            push!(kernel_ranks, r)

            # ---------------------------------------------------------
            # Relative error
            # ---------------------------------------------------------
            K_approx =
                (Utilde * Sr) *
                transpose(Vtilde)

            K_diff = copy(K_actual)

            axpy!(
                -1.0,
                K_approx,
                K_diff
            )

            rel_error =
                norm(K_diff) /
                norm_K_actual

            push!(
                kernel_errors,
                rel_error
            )

            println("Rank = ", r)

            @printf(
                "Relative Error = %.6e\n",
                rel_error
            )
        end

        # ==========================================================
        # Add kernel column
        # ==========================================================

        df_error[!, kernel_choice] =
            kernel_errors

        df_rank[!, kernel_choice] =
            kernel_ranks
    end

    # ==========================================================
    # Save CSV files
    # ==========================================================

    output_error = joinpath(
        @__DIR__,
        "..", "..",
        "OUTPUTS",
        "other_experiments_results",
        "N_vs_error.csv"
    )

    output_rank = joinpath(
        @__DIR__,
        "..", "..",
        "OUTPUTS",
        "other_experiments_results",
        "N_vs_rank.csv"
    )

    CSV.write(output_error, df_error)
    CSV.write(output_rank, df_rank)

    println("\n====================================================")
    println("Saved experiment data:")
    println(output_error)
    println(output_rank)
    println("====================================================")

end

main()