# ==============================
# chebyshev nodes generation and testing
# ==============================

print("\033[2J\033[H")
flush(stdout)

# ================================================================================================
# To save the terminl outputs to a file, run the script with:
# julia tests/02_complex_domains/arc_dot_domain_test.jl
# rememver that you have to be in cheb_nbd_nodes directory to run the above command
# ================================================================================================


# Load required files
include(joinpath(@__DIR__, "..", "..", "src", "kernels.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "geometry_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "LoCCA_SI_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "cheb_nbd_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "kernel_matrix_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "ppACA.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "chebyshev_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "SI_utils.jl"))

# Load required packages
using LinearAlgebra
BLAS.set_num_threads(8)  # Set the number of threads for BLAS operations to 8 (adjust as needed)
using Plots
using DelimitedFiles
using Random

# # Load config
# include(joinpath(@__DIR__, "config.jl"))
# using .Test_Config

function main()

    dim = 2 
    p = 17                             # number of chebyshev nodes per dimension
    kernel_choice = "kernel_1"
    l = 1
    tol = 1e-8

    # ====================================================================================
    # Define bounding hypercubes for the individual domains (Arc and Dot)
    # ====================================================================================
    target_hypercube = [-1.8  0.78; -1.1  1.1] # Contains Red Arc and Red Dot
    source_hypercube = [-0.78  1.8; -1.1  1.1] # Contains Blue Arc and Blue Dot

    # Arc thickness parameter
    delta = 0.35

    # -------------------------------------------------------------------------
    # Generate Chebyshev grids (only once)
    # -------------------------------------------------------------------------
    x_cheb_grid = generate_chebyshev_grid(target_hypercube, p)
    y_cheb_grid = generate_chebyshev_grid(source_hypercube, p)

    # weights for SI (only once)

    wx = chebyshev_tensor_weights(target_hypercube, p)
    wy = chebyshev_tensor_weights(source_hypercube, p)

    # -------------------------------------------------------------------------
    # Setup Loop Parameters and Data Arrays
    # -------------------------------------------------------------------------
    grid_sizes = 700:500:5200

    error_results = Any[
        ["Grid_Size" "Rank" "P_LoCCA_Error" "F_LoCCA_Error" "SVD_Error" "ppACA_Error" "SI_Error"]
    ]

    println("\n============================================================")
    println("          RUNNING ARC-DOT MULTI-GRID EXPERIMENTS")
    println("============================================================")
    println("Kernel Choice         : ", kernel_choice)
    println("Grid Type             : Arc and Interchanged Dot Domains")
    println("Chebyshev Grid Size   : ", p^dim)
    println("Nbd Size l            : ", l)
    println("Target Accuracy       : ", tol)
    println("Delta                 : ", delta)
    println("Target Hypercube      : ", target_hypercube)
    println("Source Hypercube      : ", source_hypercube)
    println("============================================================")

    for N in grid_sizes

        println("\n>>> Processing Grid Size N = ", N, " ...")

        # ---------------------------------------------------------
        # Generate Arc-Dot Domains
        # ---------------------------------------------------------
        Random.seed!(42)

        target_grid = generate_interchanged_domains(N, delta, 0)
        source_grid = generate_interchanged_domains(N, delta, 1)

        # ---------------------------------------------------------
        # Save coordinates only for N = 1700
        # ---------------------------------------------------------
        if N == 1700

            println("    [Saving Grid Coordinates for N = 1700 (4-column format)]")

            headers = ["source_x" "source_y" "target_x" "target_y"]

            combined_grid_data = hcat(
                source_grid[:, 1],
                source_grid[:, 2],
                target_grid[:, 1],
                target_grid[:, 2]
            )

            grid_points_1700 = vcat(headers, combined_grid_data)

            points_file = joinpath(
                @__DIR__,
                "..", "..",
                "OUTPUTS",
                "other_experiments_results",
                "arc_dot_grid_points_1700.csv"
            )

            writedlm(points_file, grid_points_1700, ',')
        end

        # ---------------------------------------------------------
        # Compute actual kernel matrix
        # ---------------------------------------------------------
        K_actual = get_matrix_from_grid_nodes(
            target_grid,
            source_grid,
            kernel_choice
        )

        norm_K_actual = norm(K_actual)

        # ---------------------------------------------------------
        # 1. F-LoCCA
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

        C_LoCCA = get_matrix_from_grid_nodes(
            target_grid,
            J_LoCCA,
            kernel_choice
        )

        U_LoCCA = get_matrix_from_grid_nodes(
            I_LoCCA,
            J_LoCCA,
            kernel_choice
        )

        R_LoCCA = get_matrix_from_grid_nodes(
            I_LoCCA,
            source_grid,
            kernel_choice
        )

        R_hat_LoCCA = U_LoCCA \ R_LoCCA
        K_approx_flocca = C_LoCCA * R_hat_LoCCA

        K_diff = copy(K_actual)
        axpy!(-1.0, K_approx_flocca, K_diff)

        f_locca_error = norm(K_diff) / norm_K_actual

        # ---------------------------------------------------------
        # 2. P-LoCCA
        # ---------------------------------------------------------
        Utilde, Sr, Vtilde =
            cheb_nbd_lowrank_from_construction(
                target_grid,
                source_grid,
                x_cheb_grid,
                y_cheb_grid,
                l,
                r,
                kernel_choice
            )

        temp = Utilde * Sr
        K_approx_plocca = temp * transpose(Vtilde)

        K_diff = copy(K_actual)
        axpy!(-1.0, K_approx_plocca, K_diff)

        p_locca_error = norm(K_diff) / norm_K_actual

        # ---------------------------------------------------------
        # 3. ACA
        # ---------------------------------------------------------
        I, J, _, _, _ =
            perform_ppACA_rank(
                kernel_choice,
                target_grid,
                source_grid,
                r,
                1
            )

        C_ppACA = get_matrix_from_grid_nodes(
            target_grid,
            source_grid[J, :],
            kernel_choice
        )

        U_ppACA = get_matrix_from_grid_nodes(
            target_grid[I, :],
            source_grid[J, :],
            kernel_choice
        )

        R_ppACA = get_matrix_from_grid_nodes(
            target_grid[I, :],
            source_grid,
            kernel_choice
        )

        R_hat_ppACA = U_ppACA \ R_ppACA
        K_approx_ppACA = C_ppACA * R_hat_ppACA

        K_diff = copy(K_actual)
        axpy!(-1.0, K_approx_ppACA, K_diff)

        ppACA_error = norm(K_diff) / norm_K_actual

        # ---------------------------------------------------------
        # 4. SI (Skeletonized Interpolation)
        # ---------------------------------------------------------

        Xhat, Yhat, _, _, _ =
            SI_nodes_from_grids_SRRQR(
                kernel_choice,
                x_cheb_grid,
                y_cheb_grid,
                wx,
                wy,
                r
            )

        C_si = get_matrix_from_grid_nodes(
            target_grid,
            Yhat,
            kernel_choice
        )

        U_si = get_matrix_from_grid_nodes(
            Xhat,
            Yhat,
            kernel_choice
        )

        R_si = get_matrix_from_grid_nodes(
            Xhat,
            source_grid,
            kernel_choice
        )

        R_hat_si = U_si \ R_si
        K_approx_si = C_si * R_hat_si

        K_diff = copy(K_actual)
        axpy!(-1.0, K_approx_si, K_diff)

        si_error = norm(K_diff) / norm_K_actual

        # ---------------------------------------------------------
        # 5. SVD
        # ---------------------------------------------------------
        U_svd, S_svd, V_svd = svd(K_actual)

        K_svd_approx =
            U_svd[:, 1:r] *
            Diagonal(S_svd[1:r]) *
            V_svd[:, 1:r]'

        K_diff = copy(K_actual)
        axpy!(-1.0, K_svd_approx, K_diff)

        svd_error = norm(K_diff) / norm_K_actual

        # ---------------------------------------------------------
        # Print current results
        # ---------------------------------------------------------
        println("    Rank                  = ", r)
        println("    P-LoCCA Error         = ", p_locca_error)
        println("    F-LoCCA Error         = ", f_locca_error)
        println("    SVD Error             = ", svd_error)
        println("    ACA Error             = ", ppACA_error)
        println("    SI Error              = ", si_error)

        # ---------------------------------------------------------
        # Store results
        # ---------------------------------------------------------
        push!(
            error_results,
            reshape(
                Any[
                    N,
                    r,
                    p_locca_error,
                    f_locca_error,
                    svd_error,
                    ppACA_error,
                    si_error
                ],
                1,
                :
            )
        )

        # Free memory before next iteration
        GC.gc()
    end

    # -------------------------------------------------------------------------
    # Save error table
    # -------------------------------------------------------------------------
    errors_file = joinpath(
        @__DIR__,
        "..", "..",
        "OUTPUTS",
        "other_experiments_results",
        "arc_dot_relative_errors.csv"
    )

    writedlm(errors_file, vcat(error_results...), ',')

    println("\n============================================================")
    println("EXPERIMENTS COMPLETE")
    println("Relative error trend data saved to:")
    println(errors_file)
    println("Grid coordinates for N = 1700 saved to:")
    println(joinpath(
        @__DIR__,
        "..", "..",
        "OUTPUTS",
        "other_experiments_results",
        "arc_dot_grid_points_1700.csv"
    ))
    println("============================================================\n")

end

main()