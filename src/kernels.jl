# kernels.jl

"""
    ker_fun(r, kernel_choice)

Kernel function supporting:
- Scalar r
- Vector r
- Matrix r
"""

# ---------------------------
# Scalar version
# ---------------------------
function ker_fun(r::Real, kernel_choice::String)

    kernel_op = kernel_choice == "kernel_1" ? (x -> 1 / x) :
                kernel_choice == "kernel_2" ? (x -> log(x)) :
                kernel_choice == "kernel_3" ? (x -> sin(x)) :
                kernel_choice == "kernel_4" ? (x -> cos(x) / x) :
                kernel_choice == "kernel_5" ? (x -> 1 / sqrt(1 + x)) :
                kernel_choice == "kernel_6" ? (x -> exp(-x^2)) :
                kernel_choice == "kernel_7" ? (x -> sqrt(1 + x^2)) :
                error("Unknown kernel function selected.")

    if r == 0
        return 0.0
    end

    val = kernel_op(r)
    return isfinite(val) ? val : 0.0
end

# ---------------------------
# Array version
# ---------------------------
function ker_fun(r::AbstractArray, kernel_choice::String)
    return ker_fun.(r, Ref(kernel_choice))
end



