# chebyshev nodes generation and testing
# ==============================

print("\033[2J\033[H")
flush(stdout)

# Load required files
include(joinpath(@__DIR__, "..", "..", "src", "kernels.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "geometry_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "LoCCA_SI_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "kernel_matrix_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "ppACA.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "chebyshev_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "SI_utils.jl"))

# Load required packages
using LinearAlgebra
using Random 

BLAS.set_num_threads(8)  

# Load config
include(joinpath(@__DIR__, "config.jl"))
using .Test_Config

function main()

    pms = PARAMS              # load the parameter data.

    target_hypercube = pms.target_hypercube
    source_hypercube = pms.source_hypercube

    grid_type = "linspace"
    dim = size(source_hypercube,1)
    p = pms.chev_nodes_p                          
    kernel_choice = pms.kernel_choice             
    nbd_size = pms.nbd_size                       
    grid_size= pms.num_domain_nodes
    tol = pms.tol                                   

    println("\n============================================================")
    println("                    EXPERIMENT DETAILS")
    println("============================================================")
    println("Kernel Choice         : ", kernel_choice)
    println("Grid Size             : ", grid_size^dim)
    println("Grid Type             : ", grid_type)
    println("Dimension             : ", dim)
    println("Chebyshev Grid Size   : ", p^dim)
    println("Nbd Size l            : ", nbd_size)
    println("Target accuracy       : ", tol)
    println("Target Hypercube      : ", target_hypercube)
    println("Source Hypercube      : ", source_hypercube)
    println("============================================================\n")

    XX_axes, YY_axes = generate_hypercube_axes(source_hypercube, target_hypercube, grid_size, grid_type)

    target_grid = generate_grid_points(XX_axes)
    source_grid = generate_grid_points(YY_axes)

    x_cheb_grid = generate_chebyshev_grid(target_hypercube, p)
    y_cheb_grid = generate_chebyshev_grid(source_hypercube, p)

    l = nbd_size
    num_runs = 10

    # -------------------------------------------------------------------------
    # Execution: F-LoCCA (10 Runs)
    # -------------------------------------------------------------------------
    println("\nRunning F-LoCCA for ", num_runs, " iterations...")
    I_LoCCA, J_LoCCA, r = 0, 0, 0
    total_LoCCA_time = 0.0
    
    for run in 1:num_runs
        t = @elapsed begin
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
        end
        total_LoCCA_time += t
    end
    avg_LoCCA_time = total_LoCCA_time / num_runs

    # -------------------------------------------------------------------------
    # Execution: ACA (10 Runs)
    # -------------------------------------------------------------------------
    println("Running ACA for ", num_runs, " iterations...")
    I, J = [], []
    total_ACA_time = 0.0
    
    for run in 1:num_runs
        t = @elapsed begin
            I, J, _, _ =
                perform_ppACA_rank_fast(
                    kernel_choice,
                    target_grid,
                    source_grid,
                    r,
                    1
                )
        end
        total_ACA_time += t
    end
    avg_ACA_time = total_ACA_time / num_runs

    # -------------------------------------------------------------------------
    # Execution: SI (10 Runs)
    # -------------------------------------------------------------------------
    println("Running SI for ", num_runs, " iterations...")
    Xhat, Yhat = 0, 0
    total_SI_time = 0.0
    
    for run in 1:num_runs
        t = @elapsed begin
            Xhat, Yhat, _, _, _ = 
                skeletonized_interpolation_nodes_rank(
                    kernel_choice, 
                    target_hypercube,
                    source_hypercube, 
                    p, 
                    r  
                )
        end
        total_SI_time += t
    end
    avg_SI_time = total_SI_time / num_runs

    println("\n============================================================")
    println("                    AVERAGE TIMING RESULTS (10 Runs)")
    println("============================================================")
    println("Avg Time taken by F-LoCCA = ", round(avg_LoCCA_time, digits = 4), " seconds")
    println("Avg Time taken by ACA     = ", round(avg_ACA_time, digits = 4), " seconds")
    println("Avg Time taken by SI      = ", round(avg_SI_time, digits = 4), " seconds")


    # =========================================================================
    # RELATIVE ERROR CALCULATION VIA RANDOM SAMPLING (Using nodes from last run)
    # =========================================================================
    println("\n============================================================")
    println("                      ERROR EVALUATION ")
    println("============================================================")

    println("Rank of approximated matrix = ", r)

    NN = grid_size^dim
    n_sample = min(2000, NN)  

    println("\nSampling ", n_sample, " random rows and columns for error estimation...")

    Random.seed!(42) 

    sampled_rows = randperm(NN)[1:n_sample]
    sampled_cols = randperm(NN)[1:n_sample]

    sub_target_grid = target_grid[sampled_rows, :]
    sub_source_grid = source_grid[sampled_cols, :]
    
    println("\nComputing exact submatrix...")
    K_true_sub = get_matrix_from_grid_nodes(sub_target_grid, sub_source_grid, kernel_choice)
    norm_K_true = norm(K_true_sub)

    # -------------------------------------------------------------------------
    # LoCCA Submatrix Error
    # -------------------------------------------------------------------------
    println("\nEvaluating LoCCA error...")
    
    
    C_LoCCA_sub = get_matrix_from_grid_nodes(sub_target_grid, J_LoCCA, kernel_choice)
    U_LoCCA_core= get_matrix_from_grid_nodes(I_LoCCA, J_LoCCA, kernel_choice)
    R_LoCCA_sub = get_matrix_from_grid_nodes(I_LoCCA, sub_source_grid, kernel_choice)

    R_hat_LoCCA_sub = U_LoCCA_core \ R_LoCCA_sub
    K_approx_LoCCA_sub = C_LoCCA_sub * R_hat_LoCCA_sub
    
    error_LoCCA = norm(K_true_sub - K_approx_LoCCA_sub) / norm_K_true
    println("LoCCA Relative Frobenius Error (Sampled): ", round(error_LoCCA, sigdigits=6))

    # -------------------------------------------------------------------------
    # ACA Submatrix Error
    # -------------------------------------------------------------------------
    println("\nEvaluating ACA error...")
    # println("Rank of approximated matrix = ", length(I))
    
    C_ppACA_sub = get_matrix_from_grid_nodes(sub_target_grid, source_grid[J, :], kernel_choice)
    U_ppACA     = get_matrix_from_grid_nodes(target_grid[I, :], source_grid[J, :], kernel_choice)
    R_ppACA_sub = get_matrix_from_grid_nodes(target_grid[I, :], sub_source_grid, kernel_choice)

    R_hat_ppACA_sub = U_ppACA \ R_ppACA_sub
    K_approx_ACA_sub = C_ppACA_sub * R_hat_ppACA_sub

    error_ACA = norm(K_true_sub - K_approx_ACA_sub) / norm_K_true
    println("ACA Relative Frobenius Error (Sampled):   ", round(error_ACA, sigdigits=6))

    # -------------------------------------------------------------------------
    # SI Submatrix Error
    # -------------------------------------------------------------------------
    println("\nEvaluating SI error...")
    # println("Rank of approximated matrix = ", r)
 
    C_si_sub = get_matrix_from_grid_nodes(sub_target_grid, Yhat, kernel_choice)
    U_si     = get_matrix_from_grid_nodes(Xhat, Yhat, kernel_choice)
    R_si_sub = get_matrix_from_grid_nodes(Xhat, sub_source_grid, kernel_choice)

    R_hat_si_sub = U_si \ R_si_sub
    K_approx_si_sub = C_si_sub * R_hat_si_sub

    error_SI = norm(K_true_sub - K_approx_si_sub) / norm_K_true
    println("SI Relative Frobenius Error (Sampled):    ", round(error_SI, sigdigits=6))
    println("============================================================\n")
end

main()