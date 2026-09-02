# ============================================================
# ppACA.jl
# Partially Pivoted Adaptive Cross Approximation (ppACA)
# ============================================================

using LinearAlgebra

# ------------------------------------------------------------
# Main ppACA function with kernel choice and tolerance
# ------------------------------------------------------------
function perform_ppACA(kernel_choice::String,
                       X::Matrix{<:Real},
                       Y::Matrix{<:Real},
                       tolerance::Real,
                       i_k1::Int)

    row = size(X, 1)
    col = size(Y, 1)

    k = 0
    S = zeros(Float64, row, col)

    U = zeros(Float64, row, 0)
    V = zeros(Float64, col, 0)

    I = Int[]
    J = Int[]

    while k < min(row, col)

        # ----------------------------------------------------
        # Pivot Row
        # ----------------------------------------------------
        a = get_matrix_from_grid_nodes(
                reshape(X[i_k1, :], 1, :), Y, kernel_choice)

        a = vec(a')  # ensure column vector

        a = get_vec_sum(a, k, i_k1, U, V)

        j_k1 = get_max_index(a, J)

        if abs(a[j_k1]) < eps()
            break
        end

        v_new = a ./ a[j_k1]

        # ----------------------------------------------------
        # Pivot Column
        # ----------------------------------------------------
        a_col = get_matrix_from_grid_nodes(
                    X,
                    reshape(Y[j_k1, :], 1, :),
                    kernel_choice)

        a_col = vec(a_col)

        u_new = get_vec_sum(a_col, k, j_k1, V, U)

        # ----------------------------------------------------
        # Stopping Criterion
        # ----------------------------------------------------
        if norm(u_new) * norm(v_new) <=
           tolerance * get_norm_S(S, k+1, U, V, u_new, v_new)
            break
        end

        # Update approximation
        S .+= u_new * v_new'

        # Store new columns
        U = hcat(U, u_new)
        V = hcat(V, v_new)

        push!(I, i_k1)
        push!(J, j_k1)

        # Next pivot row
        i_k1 = get_max_index(u_new, I)

        k += 1
    end

    return I, J, U, V, S
end





# ------------------------------------------------------------
# ppACA function with rank limit instead of tolerance
# ------------------------------------------------------------


function perform_ppACA_rank(kernel_choice::String,
                            X::Matrix{<:Real},
                            Y::Matrix{<:Real},
                            r_max::Int,
                            i_k1::Int)

    row = size(X, 1)
    col = size(Y, 1)

    max_rank = min(min(row, col), r_max)

    k = 0
    S = zeros(Float64, row, col)

    U = zeros(Float64, row, 0)
    V = zeros(Float64, col, 0)

    I = Int[]
    J = Int[]

    while k < max_rank

        # ----------------------------------------------------
        # Pivot Row
        # ----------------------------------------------------
        a = get_matrix_from_grid_nodes(
                reshape(X[i_k1, :], 1, :), Y, kernel_choice)

        a = vec(a')  # ensure column vector

        a = get_vec_sum(a, k, i_k1, U, V)

        j_k1 = get_max_index(a, J)

        # if pivot numerically zero → stop
        if abs(a[j_k1]) < eps()
            println("Pivot too small. Stopping early.")
            break
        end

        v_new = a ./ a[j_k1]

        # ----------------------------------------------------
        # Pivot Column
        # ----------------------------------------------------
        a_col = get_matrix_from_grid_nodes(
                    X,
                    reshape(Y[j_k1, :], 1, :),
                    kernel_choice)

        a_col = vec(a_col)

        u_new = get_vec_sum(a_col, k, j_k1, V, U)

        # ----------------------------------------------------
        # Update approximation
        # ----------------------------------------------------
        S .+= u_new * v_new'

        U = hcat(U, u_new)
        V = hcat(V, v_new)

        push!(I, i_k1)
        push!(J, j_k1)

        # Next pivot row
        i_k1 = get_max_index(u_new, I)

        k += 1
    end

    return I, J, U, V, S
end




function get_vec_sum(a, k, index_k1, A, B)

    if k == 0
        return a
    end

    correction = zeros(length(a))

    for l in 1:k
        correction .+= A[index_k1, l] .* B[:, l]
    end

    return a .- correction
end



function get_max_index(vec::AbstractVector,
                       previous_indices::Vector{Int})

    max_val = -Inf
    max_ind = 0

    for i in eachindex(vec)

        if i in previous_indices
            continue
        end

        val = abs(vec[i])

        if val > max_val
            max_val = val
            max_ind = i
        end
    end

    return max_ind
end



function get_norm_S(S, k, U, V, u_new, v_new)

    # Use squared Frobenius norm
    norm_S_squared = norm(S)^2

    norm_U_k = norm(u_new)
    norm_V_k = norm(v_new)

    value = norm_S_squared + (norm_U_k^2) * (norm_V_k^2)

    for j in 1:k-1
        inner_product = dot(U[:, j], u_new)
        value += 2 * inner_product^2
    end

    return sqrt(value)
end


function perform_ppACA_fast(kernel_choice::String,
                       X::Matrix{<:Real},
                       Y::Matrix{<:Real},
                       tolerance::Real,
                       i_k1::Int;
                       max_rank_limit::Int = 300) # Safeguard limit

    row = size(X, 1)
    col = size(Y, 1)

    # Pre-allocate dynamically growing arrays to prevent expensive hcats
    max_k = min(min(row, col), max_rank_limit)
    U = zeros(Float64, row, max_k)
    V = zeros(Float64, col, max_k)

    I = Int[]
    J = Int[]
    
    k = 0
    norm_S_sq = 0.0  # Track squared Frobenius norm of S algebraically

    while k < max_k
        # ----------------------------------------------------
        # Pivot Row (Use @views to avoid allocations)
        # ----------------------------------------------------
        @views row_nodes = reshape(X[i_k1, :], 1, :)
        a = vec(get_matrix_from_grid_nodes(row_nodes, Y, kernel_choice))

        # Vectorized subtracted residual
        if k > 0
            # a .- U[i_k1, 1:k] * V[:, 1:k]'
            @views mul!(a, V[:, 1:k], U[i_k1, 1:k], -1.0, 1.0)
        end

        j_k1 = get_max_index(a, J)

        if abs(a[j_k1]) < 1e-15
            break
        end

        # Generate v_new directly into preallocated matrix
        k += 1
        @views V[:, k] .= a ./ a[j_k1]

        # ----------------------------------------------------
        # Pivot Column
        # ----------------------------------------------------
        @views col_nodes = reshape(Y[j_k1, :], 1, :)
        a_col = vec(get_matrix_from_grid_nodes(X, col_nodes, kernel_choice))

        if k > 1
            # a_col .- U[:, 1:k-1] * V[j_k1, 1:k-1]'
            @views mul!(a_col, U[:, 1:k-1], V[j_k1, 1:k-1], -1.0, 1.0)
        end
        
        @views U[:, k] .= a_col

        # ----------------------------------------------------
        # Update Frobenius Norm Algebraically (No full S matrix!)
        # ----------------------------------------------------
        @views u_new = U[:, k]
        @views v_new = V[:, k]
        
        norm_u_sq = dot(u_new, u_new)
        norm_v_sq = dot(v_new, v_new)
        
        cross_term = 0.0
        if k > 1
            # Combined loop for cross-product inner updates
            for j in 1:(k-1)
                @views cross_term += dot(U[:, j], u_new) * dot(V[:, j], v_new)
            end
        end
        
        # New squared norm using expansion: ||S + uv'||^2 = ||S||^2 + ||u||^2||v||^2 + 2 sum( (U_j, u)(V_j, v) )
        norm_S_sq += norm_u_sq * norm_v_sq + 2.0 * cross_term
        norm_S = sqrt(max(0.0, norm_S_sq))

        # ----------------------------------------------------
        # Stopping Criterion
        # ----------------------------------------------------
        if sqrt(norm_u_sq) * sqrt(norm_v_sq) <= tolerance * norm_S
            # Step back if converged
            k -= 1
            break
        end

        push!(I, i_k1)
        push!(J, j_k1)

        # Next pivot row
        i_k1 = get_max_index(u_new, I)
        if i_k1 == 0
            break
        end
    end

    # Return trimmed views or copies of the exact rank reached
    return I, J, U[:, 1:k], V[:, 1:k]
end

# Optimized index selection tracking
function get_max_index(vec::AbstractVector, previous_indices::Vector{Int})
    max_val = -1.0
    max_ind = 0
    # Use a basic boolean mask or lookup if previous_indices grows large, 
    # but for typical small rank k, linear scan or simple lookup is fine.
    for i in eachindex(vec)
        if i in previous_indices
            continue
        end
        val = abs(vec[i])
        if val > max_val
            max_val = val
            max_ind = i
        end
    end
    return max_ind
end


function perform_ppACA_rank_fast(kernel_choice::String,
                                 X::Matrix{<:Real},
                                 Y::Matrix{<:Real},
                                 r_max::Int,
                                 i_k1::Int)

    row = size(X, 1)
    col = size(Y, 1)

    # Pre-allocate matrices to prevent expensive dynamic hcat allocations
    max_k = min(min(row, col), r_max)
    U = zeros(Float64, row, max_k)
    V = zeros(Float64, col, max_k)

    I = Int[]
    J = Int[]

    k = 0

    while k < max_k
        # ----------------------------------------------------
        # Pivot Row (Use @views to avoid slicing allocations)
        # ----------------------------------------------------
        @views row_nodes = reshape(X[i_k1, :], 1, :)
        a = vec(get_matrix_from_grid_nodes(row_nodes, Y, kernel_choice))

        # Fast in-place vectorized subtracted residual using BLAS/LAPACK mul!
        if k > 0
            # Computes: a .- V[:, 1:k] * U[i_k1, 1:k]
            @views mul!(a, V[:, 1:k], U[i_k1, 1:k], -1.0, 1.0)
        end

        j_k1 = get_max_index(a, J)

        # If pivot is numerically zero → stop early
        if abs(a[j_k1]) < 1e-15
            break
        end

        # Generate v_new directly into preallocated matrix
        k += 1
        @views V[:, k] .= a ./ a[j_k1]

        # ----------------------------------------------------
        # Pivot Column
        # ----------------------------------------------------
        @views col_nodes = reshape(Y[j_k1, :], 1, :)
        a_col = vec(get_matrix_from_grid_nodes(X, col_nodes, kernel_choice))

        if k > 1
            # Computes: a_col .- U[:, 1:k-1] * V[j_k1, 1:k-1]
            @views mul!(a_col, U[:, 1:k-1], V[j_k1, 1:k-1], -1.0, 1.0)
        end
        
        @views U[:, k] .= a_col

        # ----------------------------------------------------
        # Update Indices Matrix Trackers
        # ----------------------------------------------------
        push!(I, i_k1)
        push!(J, j_k1)

        # Next pivot row select
        @views u_new = U[:, k]
        i_k1 = get_max_index(u_new, I)
        
        if i_k1 == 0
            break
        end
    end

    # Return trimmed views matching the precise rank k reached
    return I, J, U[:, 1:k], V[:, 1:k]
end

# Optimized index selection tracking (Allocation-Free)
function get_max_index(vec::AbstractVector, previous_indices::Vector{Int})
    max_val = -1.0
    max_ind = 0
    for i in eachindex(vec)
        if i in previous_indices
            continue
        end
        val = abs(vec[i])
        if val > max_val
            max_val = val
            max_ind = i
        end
    end
    return max_ind
end