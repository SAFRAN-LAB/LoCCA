module Test_Config

export PARAMS

"""
Store all parameters in one structre.
So that it can be easily modified and pass around.
"""
struct Params
    kernel_choice::String
    num_domain_nodes::Int
    num_test::Int
    chev_nodes_p::Int
    nbd_size::Int
    tol::Real
    target_hypercube::Matrix{Float64}
    source_hypercube::Matrix{Float64}
end

const PARAMS = Params(
    "kernel_1",                 # 'kernel_choice' from the list of from kernel_1 to kernel_7 (see kernels.jl)
    65,                       # 'num_domain_nodes' in each axis to test the approximation after getting the nodes 
    2,                        # 'num_test' for the random_domain_test to generate the error_summary
    7,                          # 'chev_nodes_p' for the chebyshev nodes
    1,                          # 'nbd_size' for number of nodes to consider in the neighborhood of each chebyshev node during optimization)
    1e-9,                       # 'tol' for the low-rank approximation
    [-3.0 -1.0;  0.0  2.0],     # target_hypercube
    [1.0  3.0;  0.0  2.0]       # source_hypercube
    # [0.0 1.0; 0.0 1.0],         # target_hypercube
    # [1.0 2.0; 1.0 2.0]          # source_hypercube
)

end