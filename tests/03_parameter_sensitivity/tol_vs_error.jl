
print("\033[2J\033[H")
flush(stdout)

# =============================================================================================
# To save the terminl outputs to a file, run the script with:
# julia -t 8 tests/tol_vs_error.jl 
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
    tol_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,
                 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
    

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
    # Uniform grid setup (done only once)
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

        K_actual = get_matrix_from_grid_nodes(
            target_grid,
            source_grid,
            kernel_choice
        )

        norm_K_actual = norm(K_actual)

        kernel_errors = Float64[]

        # ---------------------------------------------------------
        # Loop over tolerances
        # ---------------------------------------------------------
        for tol in tol_list

            println("Tolerance = ", tol)

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

            K_approx =
                (Utilde * Sr) * transpose(Vtilde)

            K_diff = copy(K_actual)
            axpy!(-1.0, K_approx, K_diff)

            err_locca =
                norm(K_diff) / norm_K_actual

            push!(kernel_errors, err_locca)

            println("rank = ", r)
            @printf("error = %.6e\n", err_locca)
        end

        # Add kernel column
        df[!, kernel_choice] = kernel_errors
    end

    # ==========================================================
    # Save CSV
    # ==========================================================

    output_file = joinpath(
        @__DIR__,
        "..", "..",
        "OUTPUTS",
        "other_experiments_results",
        "tol_vs_error_all_kernels.csv"
    )

    CSV.write(output_file, df)

    println("\nSaved experiment data to:")
    println(output_file)

end

main()