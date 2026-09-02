# ==========================================
# geometry_utils.jl
# Hypercube + Grid generation utilities
# ==========================================

using Random
using Plots
gr()

# --------------------------------------------------
# Generate Hypercube Axes
# --------------------------------------------------

# function generate_hypercube_axes(
#         source_cube::Matrix{Float64},
#         target_cube::Matrix{Float64},
#         N::Int,
#         nodeType::String)
function generate_hypercube_axes(
        source_cube::Matrix{Float64},
        target_cube::Matrix{Float64},
        N::Int,
        nodeType::String)
        # rng::AbstractRNG = Random.default_rng())

    dim = size(source_cube,1)

    X = zeros(Float64, dim, N)
    Y = zeros(Float64, dim, N)

    # println("Node type: ", nodeType)

    # println("X = ", target_cube)
    # println("Y = ", source_cube)

    for d in 1:dim

        x0, x1 = target_cube[d,1], target_cube[d,2]
        y0, y1 = source_cube[d,1], source_cube[d,2]

        if nodeType == "linspace"

            X[d,:] .= range(x0, x1; length=N)
            Y[d,:] .= range(y0, y1; length=N)

        elseif nodeType == "random"

            X[d,:] .= x0 .+ (x1-x0).*rand(N)
            Y[d,:] .= y0 .+ (y1-y0).*rand(N)

        else
            error("Unknown nodeType")
        end
    end

    return X, Y
end





# --------------------------------------------------
# Generate Full Tensor Grid Points
# coords: (dim × N)
# returns: (N^dim × dim)
# --------------------------------------------------
function generate_grid_points(coords::Matrix{Float64})

    dim, n_points = size(coords)

    if dim == 1
        return reshape(coords, :, 1)
    end

    # Create tuple of coordinate vectors
    coord_vectors = [coords[d, :] for d in 1:dim]

    # Use Iterators.product (Julia equivalent of ndgrid)
    grid_iter = Iterators.product(coord_vectors...)

    # Collect into matrix
    grid_points = zeros(Float64, n_points^dim, dim)

    idx = 1
    for point in grid_iter
        for d in 1:dim
            grid_points[idx, d] = point[d]
        end
        idx += 1
    end

    return grid_points
end



# ------------------------------------------------------------
# Generate structured 2D initial guesses
# domain = [xmin xmax; ymin ymax]
# ------------------------------------------------------------
function initial_guesses_2D(domain::Matrix, n::Int; shuffle::Bool=false)

    xmin, xmax = domain[1,1], domain[1,2]
    ymin, ymax = domain[2,1], domain[2,2]

    hx = (xmax - xmin) / n
    hy = (ymax - ymin) / n

    xc = xmin .+ hx .* ((1:n) .- 0.5)
    yc = ymin .+ hy .* ((1:n) .- 0.5)

    # meshgrid equivalent
    X = repeat(xc', n, 1)
    Y = repeat(yc, 1, n)

    X0 = hcat(vec(X), vec(Y))

    if shuffle
        idx = randperm(size(X0,1))
        X0 = X0[idx, :]
    end

    return X0
end



# ------------------------------------------------------------
# Generate scatter plot of t and s nodes in their domains
# t_nodes, s_nodes: (r × 2) matrices of optimal nodes
# ------------------------------------------------------------

using Plots
gr()

function plot_ts_nodes(t_nodes::AbstractMatrix,
                       s_nodes::AbstractMatrix,
                       target_cube::Matrix{Float64},
                       source_cube::Matrix{Float64})

    r = size(t_nodes, 1)

    # Extract bounds
    tx0, tx1 = target_cube[1,1], target_cube[1,2]
    ty0, ty1 = target_cube[2,1], target_cube[2,2]

    sx0, sx1 = source_cube[1,1], source_cube[1,2]
    sy0, sy1 = source_cube[2,1], source_cube[2,2]

    # Plot limits with small padding
    xmin = min(tx0, sx0) - 0.5
    xmax = max(tx1, sx1) + 0.5
    ymin = min(ty0, sy0) - 0.2
    ymax = max(ty1, sy1) + 0.2

    plt = plot(
        aspect_ratio = :equal,
        grid = true,
        xlabel = "x",
        ylabel = "y",
        title = "t-nodes and s-nodes in their domains",
        xlim = (xmin, xmax),
        ylim = (ymin, ymax),
        legend = :outertop,
        size = (700, 500)
    )

    # --------------------------------------------------
    # Target domain rectangle
    # --------------------------------------------------

    plot!(plt,
          Shape([tx0, tx1, tx1, tx0],
                [ty0, ty0, ty1, ty1]),
          linecolor = :black,
          linewidth = 2,
          fillalpha = 0,
          label = "target domain")

    # --------------------------------------------------
    # Source domain rectangle
    # --------------------------------------------------

    plot!(plt,
          Shape([sx0, sx1, sx1, sx0],
                [sy0, sy0, sy1, sy1]),
          linecolor = :black,
          linestyle = :dash,
          linewidth = 2,
          fillalpha = 0,
          label = "source domain")

    # --------------------------------------------------
    # Plot nodes
    # --------------------------------------------------

    scatter!(plt,
             t_nodes[:,1],
             t_nodes[:,2],
             markersize = 7,
             markercolor = :red,
             markerstrokecolor = :black,
             marker = :circle,
             label = "t-nodes")

    scatter!(plt,
             s_nodes[:,1],
             s_nodes[:,2],
             markersize = 7,
             markercolor = :blue,
             markerstrokecolor = :black,
             marker = :square,
             label = "s-nodes")

    return plt
end



function hypercube_to_bounds(target_cube::Matrix{Float64},
                             source_cube::Matrix{Float64})

    dim_t = size(target_cube,1)
    dim_s = size(source_cube,1)

    if dim_t != dim_s
        error("Source and target hypercubes must have same dimension")
    end

    lower = vcat(target_cube[:,1], source_cube[:,1])
    upper = vcat(target_cube[:,2], source_cube[:,2])

    return lower, upper
end


# ------------------------------------------------------------
# The following funcion generates points so that it look like a smiley face in the box domains.
# domain_type = 0 for Domain X (smile), domain_type = 1 for Domain Y (frown)
# ------------------------------------------------------------


function generate_single_domain_features(N_domain::Int, delta::Float64, domain_type::Int)
    # Define features and their relative statistical weights (probabilities)
    # Component order: 1:Bottom, 2:Top, 3:Left, 4:Right, 5:LeftEye, 6:RightEye, 7:Mouth
    feature_weights = [0.18, 0.18, 0.18, 0.18, 0.05, 0.05, 0.20]
    feature_intervals = cumsum(feature_weights)

    x_cand = Vector{Float64}(undef, N_domain)
    y_cand = Vector{Float64}(undef, N_domain)
    
    # Setup boundary based on domain_type (0 for Domain X, 1 for Domain Y)
    left_bound = domain_type == 0 ? -3.0 : 1.0
    right_bound = domain_type == 0 ? -1.0 : 3.0
    
    for idx in 1:N_domain
        # Select feature based on variable weights
        r = rand()
        feature = findfirst(x -> x >= r, feature_intervals)
        
        # Inward thickness displacement variable
        thick_inward = rand() * delta
        
        if feature == 1       # Bottom Wall (Extends UPWARD into the box)
            x_cand[idx] = rand() * 2.0 + left_bound
            y_cand[idx] = 0.0 + thick_inward
        elseif feature == 2   # Top Wall (Extends DOWNWARD into the box)
            x_cand[idx] = rand() * 2.0 + left_bound
            y_cand[idx] = 2.0 - thick_inward
        elseif feature == 3   # Left Wall (Extends RIGHTWARD into the box)
            x_cand[idx] = left_bound + thick_inward
            y_cand[idx] = rand() * 2.0
        elseif feature == 4   # Right Wall (Extends LEFTWARD into the box)
            x_cand[idx] = right_bound - thick_inward
            y_cand[idx] = rand() * 2.0
            
        # Eyes (Expanded solid circular dots)
        elseif feature == 5   # Left Eye
            ex = left_bound + 0.6
            ey = domain_type == 0 ? 1.3 : 0.755
            
            eye_radius = 0.12  
            r_eye = eye_radius * sqrt(rand())
            theta_eye = rand() * 2 * pi
            x_cand[idx] = ex + r_eye * cos(theta_eye)
            y_cand[idx] = ey + r_eye * sin(theta_eye)
            
        elseif feature == 6   # Right Eye
            ex = left_bound + 1.4
            ey = domain_type == 0 ? 1.3 : 0.755
            
            eye_radius = 0.12  
            r_eye = eye_radius * sqrt(rand())
            theta_eye = rand() * 2 * pi
            x_cand[idx] = ex + r_eye * cos(theta_eye)
            y_cand[idx] = ey + r_eye * sin(theta_eye)
            
        # Mouth Arc (Radius = 0.65, centered thickness spread)
        elseif feature == 7
            cx, cy = left_bound + 1.0, 1.0
            thick_shift = (rand() - 0.5) * delta
            r_mouth = 0.65 + thick_shift
            
            if domain_type == 0
                # Domain X: Smile (Bottom semi-circle arc)
                theta = rand() * pi - pi 
                x_cand[idx] = cx + r_mouth * cos(theta)
                y_cand[idx] = cy + r_mouth * sin(theta)
            else
                # Domain Y: Frown (Top semi-circle arc)
                theta = rand() * pi
                x_cand[idx] = cx + r_mouth * cos(theta)
                y_cand[idx] = cy + r_mouth * sin(theta)
            end
        end
    end
    
    return [x_cand y_cand]
end




# ------------------------------------------------------------
# The following function generates points that look like an arc and a dot, but for source and target domains the arc and dot are interchanged.
# domain_type = 0 for Source (Arc on Left, Dot on Right), domain_type = 1 for Target (Arc on Right, Dot on Left)
# ------------------------------------------------------------  



function generate_interchanged_domains(N_domain::Int, delta::Float64, domain_type::Int)
    # domain_type == 0 -> Source (Blue components)
    # domain_type == 1 -> Target (Red components)
    
    x_cand = Vector{Float64}(undef, N_domain)
    y_cand = Vector{Float64}(undef, N_domain)
    
    cy = 0.0
    
    # 70% of points assigned to the Arc, 30% assigned to the Central Dot
    N_arc = round(Int, 0.7 * N_domain)
    
    for idx in 1:N_domain
        if idx <= N_arc
            # --- FEATURE 1: THICK CURVED ARC ---
            base_radius = 1.1
            r_arc = base_radius + (rand() - 0.5) * delta
            
            if domain_type == 0
                # Source: Left Arc '(' positioned at x = -1.6
                cx_arc = -0.5
                theta = (2/3 * π) + rand() * (2/3 * π)
                x_cand[idx] = cx_arc + r_arc * cos(theta)
            else
                # Target: Right Arc ')' positioned at x = 1.6
                cx_arc = 0.5
                theta = (-1/3 * π) + rand() * (2/3 * π)
                x_cand[idx] = cx_arc + r_arc * cos(theta)
            end
            
            y_cand[idx] = cy + r_arc * sin(theta)
            
        else
            # --- FEATURE 2: CENTRAL SOLID DOT (INTERCHANGED) ---
            dot_radius = 0.22
            r_dot = dot_radius * sqrt(rand())
            theta_dot = rand() * 2 * π
            
            if domain_type == 0
                # Source (Blue) Dot is placed on the RIGHT side of the center line
                cx_dot = 0.5  # 0.85 was earlier
            else
                # Target (Red) Dot is placed on the LEFT side of the center line
                cx_dot = -0.5 # -0.85 was earlier
            end
            
            x_cand[idx] = cx_dot + r_dot * cos(theta_dot)
            y_cand[idx] = cy + r_dot * sin(theta_dot)
        end
    end
    
    return [x_cand y_cand]
end





# ------------------------------------------------------------
# The following function generates points that look like concentric annular domains, where the inner disk
# is the target and the outer ring is the source.
# ------------------------------------------------------------  




function generate_concentric_domains(N_domain::Int, domain_type::Int)
    # domain_type == 0 -> Target (Central Blue Core)
    # domain_type == 1 -> Source (Outer Red Annulus)
    
    x_cand = Vector{Float64}(undef, N_domain)
    y_cand = Vector{Float64}(undef, N_domain)
    
    # Shared center for both domains
    cx = 0.0
    cy = 0.0
    
    # Geometric parameters
    r_target_max = 0.5   # Radius of the central blue core
    r_ring_in    = 0.8   # Inner radius of the red ring (creates a gap of 0.3)
    r_ring_out   = 1.2  # Outer radius of the red ring
    
    for idx in 1:N_domain
        theta = rand() * 2 * π
        
        if domain_type == 0
            # --- TARGET DOMAIN: Central Solid Disk (Blue) ---
            # sqrt(rand()) ensures uniform spatial distribution across the area
            r = r_target_max * sqrt(rand())
            
            x_cand[idx] = cx + r * cos(theta)
            y_cand[idx] = cy + r * sin(theta)
            
        else
            # --- SOURCE DOMAIN: Outer Annular Ring (Red) ---
            # Uniform area sampling between r_ring_in and r_ring_out
            r = sqrt(r_ring_in^2 + (r_ring_out^2 - r_ring_in^2) * rand())
            
            x_cand[idx] = cx + r * cos(theta)
            y_cand[idx] = cy + r * sin(theta)
        end
    end
    
    return [x_cand y_cand]
end
