using Plots


# -------------------------------------------------------------
# 1. Helper Functions
# -------------------------------------------------------------

# Runge's function
f(x) = 1.0 / (1.0 + 25.0 * x^2)

# Unperturbed Chebyshev nodes of the first kind
function chebyshev_nodes(p)
    return [cos((2i - 1) * π / (2p)) for i in 1:p]
end

# Compute Barycentric weights for arbitrary distinct nodes
function barycentric_weights(nodes)
    p = length(nodes)
    w = ones(Float64, p)
    for j in 1:p
        for k in 1:p
            if k != j
                w[j] *= (nodes[j] - nodes[k])
            end
        end
        w[j] = 1.0 / w[j]
    end
    # Normalize weights to prevent overflow/underflow
    return w ./ maximum(abs.(w))
end

# Evaluate Barycentric Lagrange Interpolant at point x
function evaluate_barycentric(x, nodes, values, w)
    # Check if x coincides with one of the nodes
    idx = findfirst(n -> abs(x - n) < 1e-14, nodes)
    if idx !== nothing
        return values[idx]
    end
    
    num = 0.0
    den = 0.0
    for j in 1:length(nodes)
        term = w[j] / (x - nodes[j])
        num += term * values[j]
        den += term
    end
    return num / den
end

# Vectorized barycentric evaluation
function interpolate(x_grid, nodes, values)
    w = barycentric_weights(nodes)
    return [evaluate_barycentric(x, nodes, values, w) for x in x_grid]
end

# -------------------------------------------------------------
# 2. Main Experiment Setup
# -------------------------------------------------------------

p = 15 # Polynomial degree / number of nodes
x_dense = range(-1.0, 1.0, length=1000)
f_exact = f.(x_dense)

# A. Equidistant Nodes
nodes_eq = collect(range(-1.0, 1.0, length=p))
f_eq_interp = interpolate(x_dense, nodes_eq, f.(nodes_eq))

# B. Standard Chebyshev Nodes
nodes_cheb = chebyshev_nodes(p)
f_cheb_interp = interpolate(x_dense, nodes_cheb, f.(nodes_cheb))

# C. Perturbed Chebyshev Nodes
m = minimum([abs(nodes_cheb[j] - nodes_cheb[k]) for j in 1:p, k in 1:p if j != k])
delta = m / 4.0 # Admissible perturbation threshold

rng_pert = [(-1)^i * 0.8 * delta for i in 1:p] 
nodes_pert = clamp.(nodes_cheb .+ rng_pert, -1.0, 1.0)

f_pert_interp = interpolate(x_dense, nodes_pert, f.(nodes_pert))

# -------------------------------------------------------------
# 3a. Visualization 1: Comparison with Equidistant
# -------------------------------------------------------------

p1 = plot(x_dense, f_exact, label="Exact f(x)", lw=2, color=:black,
          title="Interpolation of Runge Function (p = $p)",
          xlabel="x", ylabel="f(x)", legend=:topright) 

plot!(p1, x_dense, f_eq_interp, label="Equidistant (Runge Phenomenon)", lw=1.5, ls=:dash, color=:red)
plot!(p1, x_dense, f_cheb_interp, label="Standard Chebyshev", lw=2, color=:blue)
plot!(p1, x_dense, f_pert_interp, label="Perturbed Chebyshev (δ = $(round(delta, digits=4)))", lw=2, ls=:dot, color=:green)

scatter!(p1, nodes_pert, f.(nodes_pert), label="Perturbed Nodes", color=:green, ms=4)

display(p1)
savefig(p1, "runge_interpolation_comparison.pdf")

# -------------------------------------------------------------
# 3b. Visualization 2: Standard vs. Perturbed Chebyshev (3-Subplot Zoomed View)
# -------------------------------------------------------------

# Left Subplot: Zoom on left boundary [-1.0, -0.7]
p_left = plot(x_dense, f_exact, label="Exact f(x)", lw=2, color=:black,
              xlims=(-1.0, -0.7), ylims=(-0.05, 0.25),
              title="Left Boundary Zoom", xlabel="x", ylabel="f(x)", legend=false)
plot!(p_left, x_dense, f_cheb_interp, label="Standard Chebyshev", lw=2, color=:blue)
plot!(p_left, x_dense, f_pert_interp, label="Perturbed Chebyshev", lw=2, ls=:dash, color=:green)
scatter!(p_left, nodes_cheb, f.(nodes_cheb), label="Chebyshev Nodes", color=:blue, ms=4)
scatter!(p_left, nodes_pert, f.(nodes_pert), label="Perturbed Nodes", color=:green, ms=4)

# Middle Subplot: Full Domain [-1.0, 1.0]
p_mid = plot(x_dense, f_exact, label="Exact f(x)", lw=2, color=:black,
             xlims=(-1.0, 1.0),
             title="Full Domain (p = $p)", xlabel="x", legend=false)
plot!(p_mid, x_dense, f_cheb_interp, label="Standard Chebyshev", lw=2, color=:blue)
plot!(p_mid, x_dense, f_pert_interp, label="Perturbed Chebyshev (δ = $(round(delta, digits=4)))", lw=2, ls=:dash, color=:green)
scatter!(p_mid, nodes_cheb, f.(nodes_cheb), label="Chebyshev Nodes", color=:blue, ms=3)
scatter!(p_mid, nodes_pert, f.(nodes_pert), label="Perturbed Nodes", color=:green, ms=3)

# Right Subplot: Zoom on right boundary [0.7, 1.0]
p_right = plot(x_dense, f_exact, label="Exact f(x)", lw=2, color=:black,
               xlims=(0.7, 1.0), ylims=(-0.05, 0.25),
               title="Right Boundary Zoom", xlabel="x", legend=:topright)
plot!(p_right, x_dense, f_cheb_interp, label="Standard Chebyshev", lw=2, color=:blue)
plot!(p_right, x_dense, f_pert_interp, label="Perturbed Chebyshev", lw=2, ls=:dash, color=:green)
scatter!(p_right, nodes_cheb, f.(nodes_cheb), label="Chebyshev Nodes", color=:blue, ms=4)
scatter!(p_right, nodes_pert, f.(nodes_pert), label="Perturbed Nodes", color=:green, ms=4)

# Combine the three subplots into a 1x3 horizontal layout
p3 = plot(p_left, p_mid, p_right, layout = (1, 3), size = (1200, 400))

display(p3)

# Save high-resolution outputs
savefig(p3, "chebyshev_vs_perturbed_zoomed.pdf")
# savefig(p3, "chebyshev_vs_perturbed_zoomed.png")

# -------------------------------------------------------------
# 4. Visualization 3: Relative Error Convergence vs Degree p
# -------------------------------------------------------------

p_degrees = 5:2:55
rel_err_eq = Float64[]
rel_err_cheb = Float64[]
rel_err_pert = Float64[]

for deg in p_degrees
    # Equidistant
    n_eq = collect(range(-1.0, 1.0, length=deg))
    y_eq = interpolate(x_dense, n_eq, f.(n_eq))
    push!(rel_err_eq, maximum(abs.(y_eq .- f_exact)) / maximum(abs.(f_exact)))
    
    # Chebyshev
    n_ch = chebyshev_nodes(deg)
    y_ch = interpolate(x_dense, n_ch, f.(n_ch))
    push!(rel_err_cheb, maximum(abs.(y_ch .- f_exact)) / maximum(abs.(f_exact)))
    
    # Perturbed Chebyshev
    m_deg = minimum([abs(n_ch[j] - n_ch[k]) for j in 1:deg, k in 1:deg if j != k])
    delta_deg = m_deg / 4.0
    pert_deg = [(-1)^i * 0.8 * delta_deg for i in 1:deg]
    n_pt = clamp.(n_ch .+ pert_deg, -1.0, 1.0)
    
    y_pt = interpolate(x_dense, n_pt, f.(n_pt))
    push!(rel_err_pert, maximum(abs.(y_pt .- f_exact)) / maximum(abs.(f_exact)))
end

p2 = plot(p_degrees, rel_err_eq, label="Equidistant", yscale=:log10, lw=2, color=:red,
          title="Relative L^∞ Error Convergence", xlabel="Degree p", ylabel="Relative Error",
          marker=:circle)
plot!(p2, p_degrees, rel_err_cheb, label="Standard Chebyshev", lw=2, color=:blue, marker=:square)
plot!(p2, p_degrees, rel_err_pert, label="Perturbed Chebyshev", lw=2, color=:green, marker=:diamond)

display(p2)
savefig(p2, "relative_error_convergence.pdf")