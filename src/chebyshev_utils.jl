# ==========================================
# chebyshev_utils.jl
# Chebyshev utilities
# ==========================================

using LinearAlgebra


# --------------------------------------------------
# Generate p Chebyshev nodes in [a,b]
# --------------------------------------------------
function chebyshev_nodes_1d(a::Float64, b::Float64, p::Int)

    # Chebyshev nodes on [-1,1]
    x = cos.((2 .* (1:p) .- 1) .* pi ./ (2p))

    # Map to [a,b]
    return (a + b)/2 .+ (b - a)/2 .* x
end


# --------------------------------------------------
# Generate full tensor Chebyshev grid
# domain: (d × 2)
# returns: (p^d × d)
# --------------------------------------------------
function generate_chebyshev_grid(domain::Matrix{Float64}, p::Int)

    dim = size(domain, 1)

    # Generate 1D Chebyshev nodes for each dimension
    coords = zeros(Float64, dim, p)

    for d in 1:dim
        a, b = domain[d,1], domain[d,2]
        coords[d, :] .= chebyshev_nodes_1d(a, b, p)
    end

    # Tensor-product grid
    return generate_grid_points(coords)
end






# --------------------------------------------------
# 1D Chebyshev quadrature weights
# first-kind nodes
# --------------------------------------------------
function chebyshev_weights_1d(p::Int)

    k = 1:p

    return (π / p) .* sin.((2 .* k .- 1) .* π ./ (2p))
end


# --------------------------------------------------
# Tensor-product weights for d-dimensional grid
# domain: (d × 2)
# returns vector of length p^d
# --------------------------------------------------
function chebyshev_tensor_weights(domain::Matrix{Float64}, p::Int)

    dim = size(domain, 1)

    w1d = chebyshev_weights_1d(p)

    # Repeat same weights in each dimension
    weight_vectors = [w1d for _ in 1:dim]

    # Tensor product using Iterators.product
    weight_iter = Iterators.product(weight_vectors...)

    weights = zeros(Float64, p^dim)

    idx = 1
    for wt in weight_iter
        weights[idx] = prod(wt)
        idx += 1
    end

    return weights
end


# --------------------------------------------------
# Construct diagonal weight matrix
# size = p^d × p^d
# --------------------------------------------------
function chebyshev_weight_matrix(domain::Matrix{Float64}, p::Int)

    weights = chebyshev_tensor_weights(domain, p)

    return Diagonal(weights)
end