
print("\033[2J\033[H")
flush(stdout)

# =============================================================================================
# To save the terminl outputs to a file, run the script with:
# julia -t 8 tests/l_vs_error_and_rank.jl 
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
BLAS.set_num_threads(8)  # Set the number of threads for BLAS operations to 1 (adjust as needed)
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

    NN  = pms.num_domain_nodes
    tol = 1.0e-8

    # ---------------------------------------------------------
    # p values
    # ---------------------------------------------------------
    p_list = [2,3,4,5,6,7,8,9,10]

    # ---------------------------------------------------------
    # l values
    # ---------------------------------------------------------
    l_list = 2 # [1,2,3,4,5,6,7,8]

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

    # ==========================================================
    # Uniform grid setup (done once)
    # ==========================================================

    node_type = "linspace"

    XX_axes, YY_axes = generate_hypercube_axes(
        source_hypercube,
        target_hypercube,
        NN,
        node_type
    )

    target_grid = generate_grid_points(XX_axes)
    source_grid = generate_grid_points(YY_axes)

    # ==========================================================
    # Create dataframes
    # ==========================================================

    df_error = DataFrame(l = l_list)
    df_rank  = DataFrame(l = l_list)

    # ==========================================================
    # Loop over kernels
    # ==========================================================

    for kernel_choice in kernel_list

        println("\n====================================================")
        println("Running kernel: ", kernel_choice)
        println("====================================================")

        # ---------------------------------------------------------
        # Build kernel matrix once
        # ---------------------------------------------------------
        K_actual = get_matrix_from_grid_nodes(
            target_grid,
            source_grid,
            kernel_choice
        )

        norm_K_actual = norm(K_actual)

        # ==========================================================
        # Loop over p
        # ==========================================================

        for p in p_list

            println("\n********************************************")
            println("Running p = ", p)
            println("********************************************")

            # ---------------------------------------------------------
            # Generate Chebyshev grids
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

            # ---------------------------------------------------------
            # Storage
            # ---------------------------------------------------------
            kernel_errors = Float64[]
            kernel_ranks  = Int[]

            # ==========================================================
            # Loop over l
            # ==========================================================

            for l in l_list

                println("--------------------------------------------")
                println("Running l = ", l)
                println("--------------------------------------------")

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
            # Add columns
            # ==========================================================

            col_name =
                "$(kernel_choice)_p$(p)"

            df_error[!, col_name] =
                kernel_errors

            df_rank[!, col_name] =
                kernel_ranks
        end
    end

    # ==========================================================
    # Save CSV files
    # ==========================================================

    output_error = joinpath(
        @__DIR__,
        "..", "..",
        "OUTPUTS",
        "other_experiments_results",
        "p_vs_error.csv"
    )

    # output_rank = joinpath(
    #     @__DIR__,
    #     "..",
    #     "OUTPUTS",
    #     "other_experiments_results",
    #     "l_vs_rank_multi_p.csv"
    # )

    CSV.write(output_error, df_error)
    # CSV.write(output_rank, df_rank)

    println("\n====================================================")
    println("Saved experiment data:")
    println(output_error)
    # println(output_rank)
    println("====================================================")

end

main()