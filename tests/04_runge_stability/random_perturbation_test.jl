using Plots
using Random
using DelimitedFiles # Required for exporting CSV files

Random.seed!(42)

# -------------------------------------------------------------
# 1. Helper Functions
# -------------------------------------------------------------

f(x) = 1.0 / (1.0 + 25.0 * x^2)

function chebyshev_nodes(p)
    return [cos((2i - 1) * π / (2p)) for i in 1:p]
end

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
    return w ./ maximum(abs.(w))
end

function evaluate_barycentric(x, nodes, values, w)
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

function interpolate(x_grid, nodes, values)
    w = barycentric_weights(nodes)
    return [evaluate_barycentric(x, nodes, values, w) for x in x_grid]
end

function generate_random_perturbed_nodes(nodes_cheb, δ; max_shift=0.8)
    p = length(nodes_cheb)
    random_shifts = [(2.0 * rand() - 1.0) * max_shift * δ for _ in 1:p]
    return clamp.(nodes_cheb .+ random_shifts, -1.0, 1.0)
end

# -------------------------------------------------------------
# 2. Main Experiment Setup (p = 25)
# -------------------------------------------------------------

p = 15
x_dense = collect(range(-1.0, 1.0, length=1000))
f_exact = f.(x_dense)

nodes_cheb = chebyshev_nodes(p)
f_cheb_interp = interpolate(x_dense, nodes_cheb, f.(nodes_cheb))

m = minimum([abs(nodes_cheb[j] - nodes_cheb[k]) for j in 1:p, k in 1:p if j != k])
delta = m / 4.0

nodes_pert_rand = generate_random_perturbed_nodes(nodes_cheb, delta; max_shift=0.8)
f_pert_interp = interpolate(x_dense, nodes_pert_rand, f.(nodes_pert_rand))

# EXPORT DATASET 1: Interpolation Curve Data for p = 25
# Columns: x, f_exact, f_cheb, f_perturbed
interp_data = [x_dense f_exact f_cheb_interp f_pert_interp]
writedlm("interp_curves_p25.csv", interp_data, ',')

# EXPORT DATASET 2: Node Positions for Scatter Plots
# Columns: cheb_nodes, f_cheb_nodes, pert_nodes, f_pert_nodes
nodes_data = [nodes_cheb f.(nodes_cheb) nodes_pert_rand f.(nodes_pert_rand)]
writedlm("interp_nodes_p25.csv", nodes_data, ',')

# -------------------------------------------------------------
# 3. Relative Error Convergence Data
# -------------------------------------------------------------

p_degrees = collect(5:2:55)
rel_err_eq = Float64[]
rel_err_cheb = Float64[]
rel_err_pert = Float64[]

for deg in p_degrees
    n_eq = collect(range(-1.0, 1.0, length=deg))
    y_eq = interpolate(x_dense, n_eq, f.(n_eq))
    push!(rel_err_eq, maximum(abs.(y_eq .- f_exact)) / maximum(abs.(f_exact)))
    
    n_ch = chebyshev_nodes(deg)
    y_ch = interpolate(x_dense, n_ch, f.(n_ch))
    push!(rel_err_cheb, maximum(abs.(y_ch .- f_exact)) / maximum(abs.(f_exact)))
    
    m_deg = minimum([abs(n_ch[j] - n_ch[k]) for j in 1:deg, k in 1:deg if j != k])
    delta_deg = m_deg / 4.0
    pert_deg = [(-1)^i * 0.8 * delta_deg for i in 1:deg]
    n_pt = clamp.(n_ch .+ pert_deg, -1.0, 1.0)
    
    y_pt = interpolate(x_dense, n_pt, f.(n_pt))
    push!(rel_err_pert, maximum(abs.(y_pt .- f_exact)) / maximum(abs.(f_exact)))
end

# EXPORT DATASET 3: Relative Error Convergence vs Degree p
# Columns: p_degree, rel_err_equidistant, rel_err_chebyshev, rel_err_perturbed
convergence_data = [p_degrees rel_err_eq rel_err_cheb rel_err_pert]
writedlm("relative_error_convergence.csv", convergence_data, ',')

println("CSV files generated successfully!")













# using Plots
# using Random

# # Set random seed for reproducibility
# Random.seed!(42)

# # -------------------------------------------------------------
# # 1. Helper Functions
# # -------------------------------------------------------------

# # Runge's function
# f(x) = 1.0 / (1.0 + 25.0 * x^2)

# # Unperturbed Chebyshev nodes of the first kind
# function chebyshev_nodes(p)
#     return [cos((2i - 1) * π / (2p)) for i in 1:p]
# end

# # Compute Barycentric weights for arbitrary distinct nodes
# function barycentric_weights(nodes)
#     p = length(nodes)
#     w = ones(Float64, p)
#     for j in 1:p
#         for k in 1:p
#             if k != j
#                 w[j] *= (nodes[j] - nodes[k])
#             end
#         end
#         w[j] = 1.0 / w[j]
#     end
#     # Normalize weights to prevent overflow/underflow
#     return w ./ maximum(abs.(w))
# end

# # Evaluate Barycentric Lagrange Interpolant at point x
# function evaluate_barycentric(x, nodes, values, w)
#     idx = findfirst(n -> abs(x - n) < 1e-14, nodes)
#     if idx !== nothing
#         return values[idx]
#     end
    
#     num = 0.0
#     den = 0.0
#     for j in 1:length(nodes)
#         term = w[j] / (x - nodes[j])
#         num += term * values[j]
#         den += term
#     end
#     return num / den
# end

# # Vectorized barycentric evaluation
# function interpolate(x_grid, nodes, values)
#     w = barycentric_weights(nodes)
#     return [evaluate_barycentric(x, nodes, values, w) for x in x_grid]
# end

# # Generate random perturbations strictly inside [-max_shift * δ, max_shift * δ]
# function generate_random_perturbed_nodes(nodes_cheb, δ; max_shift=0.8)
#     p = length(nodes_cheb)
#     # (2 * rand() - 1) produces uniformly distributed numbers in [-1, 1]
#     random_shifts = [(2.0 * rand() - 1.0) * max_shift * δ for _ in 1:p]
#     return clamp.(nodes_cheb .+ random_shifts, -1.0, 1.0)
# end

# # -------------------------------------------------------------
# # 2. Main Experiment Setup (p = 15)
# # -------------------------------------------------------------

# p = 25 # Polynomial degree
# x_dense = range(-1.0, 1.0, length=1000)
# f_exact = f.(x_dense)

# # Equidistant
# nodes_eq = collect(range(-1.0, 1.0, length=p))
# f_eq_interp = interpolate(x_dense, nodes_eq, f.(nodes_eq))

# # Chebyshev
# nodes_cheb = chebyshev_nodes(p)
# f_cheb_interp = interpolate(x_dense, nodes_cheb, f.(nodes_cheb))

# # Randomly Perturbed Chebyshev
# m = minimum([abs(nodes_cheb[j] - nodes_cheb[k]) for j in 1:p, k in 1:p if j != k])
# delta = m / 4.0 # Admissible perturbation threshold

# nodes_pert_rand = generate_random_perturbed_nodes(nodes_cheb, delta; max_shift=0.8)
# f_pert_interp = interpolate(x_dense, nodes_pert_rand, f.(nodes_pert_rand))

# # -------------------------------------------------------------
# # 3a. Visualization 1: Comparison with Equidistant
# # -------------------------------------------------------------

# p1 = plot(x_dense, f_exact, label="Exact f(x)", lw=2, color=:black,
#           title="Interpolation of Runge Function (p = $p)",
#           xlabel="x", ylabel="f(x)", legend=:topright) 

# plot!(p1, x_dense, f_eq_interp, label="Equidistant (Runge Phenomenon)", lw=1.5, ls=:dash, color=:red)
# plot!(p1, x_dense, f_cheb_interp, label="Standard Chebyshev", lw=2, color=:blue)
# plot!(p1, x_dense, f_pert_interp, label="Randomly Perturbed Chebyshev", lw=2, ls=:dot, color=:green)

# scatter!(p1, nodes_pert_rand, f.(nodes_pert_rand), label="Random Nodes", color=:green, ms=4)

# display(p1)
# savefig(p1, "runge_random_perturbation_comparison.pdf")

# # -------------------------------------------------------------
# # 3b. Visualization 2: Standard vs Randomly Perturbed (Zoomed Subplots)
# # -------------------------------------------------------------

# p_left = plot(x_dense, f_exact, label="Exact f(x)", lw=2, color=:black,
#               xlims=(-1.0, -0.7), ylims=(-0.05, 0.25),
#               title="Left Boundary Zoom", xlabel="x", ylabel="f(x)", legend=false)
# plot!(p_left, x_dense, f_cheb_interp, lw=2, color=:blue)
# plot!(p_left, x_dense, f_pert_interp, lw=2, ls=:dash, color=:green)
# scatter!(p_left, nodes_cheb, f.(nodes_cheb), color=:blue, ms=4)
# scatter!(p_left, nodes_pert_rand, f.(nodes_pert_rand), color=:green, ms=4)

# p_mid = plot(x_dense, f_exact, label="Exact f(x)", lw=2, color=:black,
#              xlims=(-1.0, 1.0),
#              title="Full Domain (p = $p)", xlabel="x", legend=false)
# plot!(p_mid, x_dense, f_cheb_interp, lw=2, color=:blue)
# plot!(p_mid, x_dense, f_pert_interp, lw=2, ls=:dash, color=:green)
# scatter!(p_mid, nodes_cheb, f.(nodes_cheb), color=:blue, ms=3)
# scatter!(p_mid, nodes_pert_rand, f.(nodes_pert_rand), color=:green, ms=3)

# p_right = plot(x_dense, f_exact, label="Exact f(x)", lw=2, color=:black,
#                xlims=(0.7, 1.0), ylims=(-0.05, 0.25),
#                title="Right Boundary Zoom", xlabel="x", legend=:topright)
# plot!(p_right, x_dense, f_cheb_interp, label="Standard Chebyshev", lw=2, color=:blue)
# plot!(p_right, x_dense, f_pert_interp, label="Random Perturbed", lw=2, ls=:dash, color=:green)
# scatter!(p_right, nodes_cheb, f.(nodes_cheb), label="Chebyshev Nodes", color=:blue, ms=4)
# scatter!(p_right, nodes_pert_rand, f.(nodes_pert_rand), label="Random Nodes", color=:green, ms=4)

# p3 = plot(p_left, p_mid, p_right, layout = (1, 3), size = (1200, 400))
# display(p3)
# savefig(p3, "random_chebyshev_vs_perturbed_zoomed.pdf")

# # -------------------------------------------------------------
# # 4. Visualization 3: Relative Error Convergence vs Degree p
# # -------------------------------------------------------------

# p_degrees = 5:2:55
# rel_err_eq = Float64[]
# rel_err_cheb = Float64[]
# rel_err_pert = Float64[]

# for deg in p_degrees
#     # Equidistant
#     n_eq = collect(range(-1.0, 1.0, length=deg))
#     y_eq = interpolate(x_dense, n_eq, f.(n_eq))
#     push!(rel_err_eq, maximum(abs.(y_eq .- f_exact)) / maximum(abs.(f_exact)))
    
#     # Chebyshev
#     n_ch = chebyshev_nodes(deg)
#     y_ch = interpolate(x_dense, n_ch, f.(n_ch))
#     push!(rel_err_cheb, maximum(abs.(y_ch .- f_exact)) / maximum(abs.(f_exact)))
    
#     # Perturbed Chebyshev
#     m_deg = minimum([abs(n_ch[j] - n_ch[k]) for j in 1:deg, k in 1:deg if j != k])
#     delta_deg = m_deg / 4.0
#     pert_deg = [(-1)^i * 0.8 * delta_deg for i in 1:deg]
#     n_pt = clamp.(n_ch .+ pert_deg, -1.0, 1.0)
    
#     y_pt = interpolate(x_dense, n_pt, f.(n_pt))
#     push!(rel_err_pert, maximum(abs.(y_pt .- f_exact)) / maximum(abs.(f_exact)))
# end

# p2 = plot(p_degrees, rel_err_eq, label="Equidistant", yscale=:log10, lw=2, color=:red,
#           title="Relative L^∞ Error Convergence", xlabel="Degree p", ylabel="Relative Error",
#           marker=:circle)
# plot!(p2, p_degrees, rel_err_cheb, label="Standard Chebyshev", lw=2, color=:blue, marker=:square)
# plot!(p2, p_degrees, rel_err_pert, label="Perturbed Chebyshev", lw=2, color=:green, marker=:diamond)

# display(p2)
# savefig(p2, "relative_error_convergence.pdf")