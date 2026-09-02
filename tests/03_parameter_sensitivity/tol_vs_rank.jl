
print("\033[2J\033[H")
flush(stdout)

# =============================================================================================
# To save the terminl outputs to a file, run the script with:
# julia -t 8 tests/tol_vs_rank.jl 
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

    NN = pms.num_domain_nodes
    p  = pms.chev_nodes_p
    l  = pms.nbd_size

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
    # Tolerances
    # ---------------------------------------------------------
    tol_list = [
        1e-2, 1e-3, 1e-4, 1e-5,
        1e-6, 1e-7, 1e-8,
        1e-9, 1e-10, 1e-11, 1e-12
    ]

    # ---------------------------------------------------------
    # Hypercube definitions
    # ---------------------------------------------------------
    target_hypercube = pms.target_hypercube
    source_hypercube = pms.source_hypercube

    # ---------------------------------------------------------
    # Generate Chebyshev grids
    # ---------------------------------------------------------
    x_cheb_grid = generate_chebyshev_grid(target_hypercube, p)
    y_cheb_grid = generate_chebyshev_grid(source_hypercube, p)

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
    # Create dataframe
    # ==========================================================

    df = DataFrame(tolerance = tol_list)

    # ==========================================================
    # Loop over kernels
    # ==========================================================

    for kernel_choice in kernel_list

        println("\n====================================================")
        println("Running kernel: ", kernel_choice)
        println("====================================================")

        # ---------------------------------------------------------
        # Build kernel matrix
        # ---------------------------------------------------------
        K_actual = get_matrix_from_grid_nodes(
            target_grid,
            source_grid,
            kernel_choice
        )

        norm_K_actual = norm(K_actual)

        # ==========================================================
        # SVD (only once per kernel)
        # ==========================================================

        println("Computing SVD...")

        U, S, V = svd(K_actual)

        singular_sq = S .^ 2

        # tail_energy[k] =
        # sum_{j=k}^end sigma_j^2
        tail_energy =
            reverse(cumsum(reverse(singular_sq)))

        n_singular = length(S)

        # ==========================================================
        # Storage
        # ==========================================================

        locca_ranks = Int[]
        svd_ranks   = Int[]

        # ==========================================================
        # Loop over tolerances
        # ==========================================================

        for tol in tol_list

            println("\nTolerance = ", tol)

            # ---------------------------------------------------------
            # LoCCA rank
            # ---------------------------------------------------------
            _, _, _, r_locca =
                cheb_nbd_lowrank_from_construction_tol(
                    target_grid,
                    source_grid,
                    x_cheb_grid,
                    y_cheb_grid,
                    l,
                    tol,
                    kernel_choice
                )

            push!(locca_ranks, r_locca)

            # ---------------------------------------------------------
            # SVD optimal rank
            # ---------------------------------------------------------
            r_svd = n_singular

            for r in 1:n_singular-1

                rel_err =
                    sqrt(tail_energy[r+1]) /
                    norm_K_actual

                if rel_err <= tol
                    r_svd = r
                    break
                end
            end

            push!(svd_ranks, r_svd)

            println("LoCCA rank = ", r_locca)
            println("SVD rank    = ", r_svd)
        end

        # ==========================================================
        # Add columns to dataframe
        # ==========================================================

        df[!, "$(kernel_choice)_LoCCA"] =
            locca_ranks

        df[!, "$(kernel_choice)_SVD"] =
            svd_ranks
    end

    # ==========================================================
    # Save CSV
    # ==========================================================

    output_file = joinpath(
        @__DIR__,
        "..", "..",
        "OUTPUTS",
        "other_experiments_results",
        "tol_vs_rank_all_kernels.csv"
    )

    CSV.write(output_file, df)

    println("\nSaved experiment data to:")
    println(output_file)

end

main()