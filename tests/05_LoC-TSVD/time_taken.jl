# ==============================
# chebyshev nodes generation and testing
# ==============================

print("\033[2J\033[H")
flush(stdout)

# Load required files
include(joinpath(@__DIR__, "..", "..", "src", "kernels.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "geometry_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "cheb_nbd_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "kernel_matrix_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "ppACA.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "chebyshev_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "SI_utils.jl"))

# Load required packages
using LinearAlgebra
using Random # <--- MOVED TO THE TOP LEVEL

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

    println("\nRunning LoCCA...")
    # FIXED: Capture Ur, Sr, and Vr instead of using underscores
    Ur, Sr, Vr, r = 0, 0, 0, 0
    LoCCA_time_taken = @elapsed begin
        Ur, Sr, Vr, r =
            cheb_nbd_lowrank_from_construction_tol(
                target_grid,
                source_grid,
                x_cheb_grid,
                y_cheb_grid,
                l,
                tol,
                kernel_choice
            )
            # LoCCA_with_grid_input(
            #     target_grid,
            #     source_grid,
            #     x_cheb_grid,
            #     y_cheb_grid,
            #     l,
            #     kernel_choice
            # )
    end

    # Vr = transpose(Vr) # this we are doing ony for the locca algo

    println("\nRunning ACA...")
    # FIXED: Capture indices I and J instead of using underscores
    I, J = [], []
    ACA_time_taken = @elapsed begin
        I, J, _, _ =
            # perform_ppACA_fast(
            #     kernel_choice,
            #     target_grid,
            #     source_grid,
            #     tol,
            #     1
            # )
            perform_ppACA_rank_fast(
                kernel_choice,
                target_grid,
                source_grid,
                r,
                1
            )
    end

    println("Time taken by LoCCA = ", round(LoCCA_time_taken, digits = 4), " seconds")
    println("Time taken by ACA   = ", round(ACA_time_taken, digits =   4), " seconds")


    # =========================================================================
    # RELATIVE ERROR CALCULATION VIA RANDOM SAMPLING
    # =========================================================================
    println("\n============================================================")
    println("                    ERROR EVALUATION")
    println("============================================================")

     NN = grid_size^dim
    # N_source = size(source_grid, 1)
    n_sample = min(2000, NN)  # Sample at most 2000 points or the total number of points if smaller

    println("Sampling ", n_sample, " random rows and columns for error estimation...")

    # Seed is safely set here now that 'using Random' is at the top
    Random.seed!(42) 

    sampled_rows = randperm(NN)[1:n_sample]
    sampled_cols = randperm(NN)[1:n_sample]

    sub_target_grid = target_grid[sampled_rows, :]
    sub_source_grid = source_grid[sampled_cols, :]
    
    println("Computing exact submatrix...")
    K_true_sub = get_matrix_from_grid_nodes(sub_target_grid, sub_source_grid, kernel_choice)
    norm_K_true = norm(K_true_sub)

    # -------------------------------------------------------------------------
    # LoCCA Submatrix Error
    # -------------------------------------------------------------------------
    println("Evaluating LoCCA error...")
    println("Rank of approximated matrix = ", r)
    
    Ur_sub = Ur[sampled_rows, :]
    Vr_sub = Vr[sampled_cols, :]  
    
    K_approx_LoCCA_sub = (Ur_sub * Sr) * transpose(Vr_sub)
    # K_approx_LoCCA_sub = Ur_sub * (Sr\ transpose(Vr_sub))   # for LoCCA only
    
    error_LoCCA = norm(K_true_sub - K_approx_LoCCA_sub) / norm_K_true
    println("LoCCA Relative Frobenius Error (Sampled): ", round(error_LoCCA, sigdigits=6))

    # -------------------------------------------------------------------------
    # ACA Submatrix Error
    # -------------------------------------------------------------------------
    println("Evaluating ACA error...")
    println("Rank of approximated matrix = ", length(I))
    
    
    C_ppACA_sub = get_matrix_from_grid_nodes(sub_target_grid, source_grid[J, :], kernel_choice)
    U_ppACA     = get_matrix_from_grid_nodes(target_grid[I, :], source_grid[J, :], kernel_choice)
    R_ppACA_sub = get_matrix_from_grid_nodes(target_grid[I, :], sub_source_grid, kernel_choice)

    R_hat_ppACA_sub = U_ppACA \ R_ppACA_sub
    K_approx_ACA_sub = C_ppACA_sub * R_hat_ppACA_sub

    error_ACA = norm(K_true_sub - K_approx_ACA_sub) / norm_K_true
    println("ACA Relative Frobenius Error (Sampled):   ", round(error_ACA, sigdigits=6))
    println("============================================================\n")
end

main()