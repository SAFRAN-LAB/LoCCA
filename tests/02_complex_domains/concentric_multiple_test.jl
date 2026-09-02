# ==============================================================================
# Chebyshev nodes generation and testing across multiple Grid Sizes
# Saves grid points ONLY for N = 1700 in 4 parallel columns
# ==============================================================================

print("\033[2J\033[H")
flush(stdout)

# Load required files
include(joinpath(@__DIR__, "..", "..", "src", "kernels.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "geometry_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "cheb_nbd_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "LoCCA_SI_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "kernel_matrix_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "ppACA.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "chebyshev_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "SI_utils.jl"))

# Load required packages
using LinearAlgebra
BLAS.set_num_threads(8)  
using Plots
using Random
using DelimitedFiles  # Required for exporting data to CSV format

# Load config
# include(joinpath(@__DIR__, "..", "..", "tests", "config.jl"))
# using .Test_Config

function main()

    dim = 2 
    p = 17                             # number of chebyshev nodes per dimension
    kernel_choice = "kernel_1"         
    l = 1         
    tol = 1e-8                        

    # Define bounding hypercubes for Concentric Domains
    target_hypercube = [-0.5  0.5;  -0.5  0.5]     
    source_hypercube = [-1.2  1.2; -1.2  1.2]

    # Generate Chebyshev grids (Single bounding boxes)
    x_cheb_grid = generate_chebyshev_grid(target_hypercube, p)
    y_cheb_grid = generate_chebyshev_grid(source_hypercube, p)

    # -------------------------------------------------------------------------
    # Setup Loop Parameters and Data Arrays
    # -------------------------------------------------------------------------
    grid_sizes = 700:500:5200  # Range: 700 to 5200 with a step of 500
    
    # Initialized explicitly as Any to handle both Strings and Floats
    error_results = Any[["Grid_Size" "Rank" "P_LoCCA_Error" "F_LoCCA_Error" "SVD_Error" "ppACA_Error" "SI_Error"]]

    println("\n============================================================")
    println("             RUNNING MULTI-GRID EXPERIMENTS")
    println("============================================================")

    for N in grid_sizes
        println("\n>>> Processing Grid Size N = ", N, " ...")

        # Generate Concentric Domain's Grids
        Random.seed!(42)    
        target_grid = generate_concentric_domains(N, 0) # Blue Target (Core)
        source_grid = generate_concentric_domains(N, 1) # Red Source (Ring)

        # CHANGED: Format and save grid points ONLY for N = 1700 into 4 side-by-side columns
        if N == 1700
            println("    [Saving Grid Coordinates for N = 1700 (4-column format)]")
            
            # Create headers
            headers = ["source_x" "source_y" "target_x" "target_y"]
            
            # Combine columns horizontally: [source_x source_y target_x target_y]
            combined_grid_data = hcat(source_grid[:, 1], source_grid[:, 2], target_grid[:, 1], target_grid[:, 2])
            
            # Merge header row with data matrix
            grid_points_1700 = vcat(headers, combined_grid_data)
            
            # Write immediately to disk
            points_file = joinpath(@__DIR__, "..", "..", "OUTPUTS", "other_experiments_results", "concentric_grid_points_1700.csv")
            writedlm(points_file, grid_points_1700, ',')
        end

        # Compute actual kernel matrix
        K_actual = get_matrix_from_grid_nodes(target_grid, source_grid, kernel_choice)
        norm_K_actual = norm(K_actual)

        # ---------------------------------------------------------
        # 1. F-LoCCA 
        # ---------------------------------------------------------
        I_LoCCA, J_LoCCA, r = LoCCA_with_grid_input(
            target_grid, source_grid, x_cheb_grid, y_cheb_grid, l, kernel_choice, tol
        )

        C_LoCCA = get_matrix_from_grid_nodes(target_grid, J_LoCCA, kernel_choice)
        U_LoCCA = get_matrix_from_grid_nodes(I_LoCCA, J_LoCCA, kernel_choice)
        R_LoCCA = get_matrix_from_grid_nodes(I_LoCCA, source_grid, kernel_choice)

        R_hat_LoCCA = U_LoCCA \ R_LoCCA
        K_approx_flocca = C_LoCCA * R_hat_LoCCA

        K_diff = copy(K_actual)
        axpy!(-1.0, K_approx_flocca, K_diff)
        f_locca_error = norm(K_diff) / norm_K_actual

        # ---------------------------------------------------------
        # 2. P-LoCCA 
        # ---------------------------------------------------------
        Utilde, Sr, Vtilde = cheb_nbd_lowrank_from_construction(
            target_grid, source_grid, x_cheb_grid, y_cheb_grid, l, r, kernel_choice
        )

        temp = Utilde * Sr
        K_approx_plocca = temp * transpose(Vtilde)

        K_diff = copy(K_actual)
        axpy!(-1.0, K_approx_plocca, K_diff)
        p_locca_error = norm(K_diff) / norm_K_actual

        # ---------------------------------------------------------
        # 3. ACA
        # ---------------------------------------------------------
        I, J, _, _, _ = perform_ppACA_rank(kernel_choice, target_grid, source_grid, r, 1)
        C_ppACA = get_matrix_from_grid_nodes(target_grid, source_grid[J, :], kernel_choice)
        U_ppACA = get_matrix_from_grid_nodes(target_grid[I, :], source_grid[J, :], kernel_choice)
        R_ppACA = get_matrix_from_grid_nodes(target_grid[I, :], source_grid, kernel_choice)

        local_R_hat_ppACA = U_ppACA \ R_ppACA
        K_approx_ppACA = C_ppACA * local_R_hat_ppACA

        K_diff = copy(K_actual)
        axpy!(-1.0, K_approx_ppACA, K_diff)
        linspace_ppACA_error = norm(K_diff) / norm_K_actual

        # ---------------------------------------------------------
        # 4. SI (Skeletonized Interpolation)
        # ---------------------------------------------------------
        wx = chebyshev_tensor_weights(target_hypercube, p)
        wy = chebyshev_tensor_weights(source_hypercube, p)

        Xhat, Yhat, _, _, _ = SI_nodes_from_grids_SRRQR(kernel_choice, x_cheb_grid, y_cheb_grid, wx, wy, r)

        C_si = get_matrix_from_grid_nodes(target_grid, Yhat, kernel_choice)
        U_si = get_matrix_from_grid_nodes(Xhat, Yhat, kernel_choice)
        R_si = get_matrix_from_grid_nodes(Xhat, source_grid, kernel_choice)

        R_hat_si = U_si \ R_si
        K_approx_si = C_si * R_hat_si

        K_diff = copy(K_actual)
        axpy!(-1.0, K_approx_si, K_diff)
        linspace_si_error = norm(K_diff) / norm_K_actual

        # ---------------------------------------------------------
        # 5. SVD
        # ---------------------------------------------------------
        U_svd, S_svd, V_svd = svd(K_actual)
        K_svd_approx = U_svd[:, 1:r] * Diagonal(S_svd[1:r]) * V_svd[:, 1:r]'

        K_diff = copy(K_actual)
        axpy!(-1.0, K_svd_approx, K_diff)
        linspace_svd_error = norm(K_diff) / norm_K_actual

        # Store calculated errors for the current N into the matrix
        push!(error_results, [N r p_locca_error f_locca_error linspace_svd_error linspace_ppACA_error linspace_si_error])
    end

    # -------------------------------------------------------------------------
    # Save errors file to disk
    # -------------------------------------------------------------------------
    errors_file = joinpath(@__DIR__, "..", "..", "OUTPUTS", "other_experiments_results", "concentric_relative_errors.csv")
    writedlm(errors_file, vcat(error_results...), ',')

    println("\n============================================================")
    println("EXPERIMENTS COMPLETE")
    println("Relative error trend data (700-5200) saved to: ", errors_file)
    println("Domain grid coordinates (Only N=1700) saved to standard 4-column destination path.")
    println("============================================================\n")
end

main()