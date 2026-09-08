# LoCCA: Localized Chebyshev Cross Approximation for Kernel Matrix Factorization

---

This repository contains the official Julia implementation for the research paper:
> **"LoCCA: Localized Chebyshev Cross Approximation for Kernel Matrix Factorization via Nodal Perturbation Stability"**  
> *Authors: Sumit Singh, Shrirup Dutta, and Sivaram Ambikasaran*

---

## 📌 Overview
Dense kernel matrices arise ubiquitously across integral equations, boundary element methods, Gaussian processes, and radial basis function approximations. Standard analytic methods (such as Skeletonized Interpolation) rely on rigid Cartesian Chebyshev grids that suffer catastrophic failure due to sample starvation on non-convex or unstructured domains. Conversely, purely algebraic cross-approximation heuristics (such as ACA) frequently encounter slow convergence, wide error spreads, and extreme outlier spikes.

**LoCCA (Localized Chebyshev Cross Approximation)** bridges continuous polynomial interpolation with data-driven matrix skeleton factorizations:
1. **Localized Node Selection (LoC-NS):** Maps continuous tensor-product Chebyshev anchors to nearby discrete physical neighbors, treating localized selections rigorously as structured perturbations of ideal grids.
2. **Low-Rank Cross-Recompression:** Extracts structure-preserving skeletons ($\tilde{\mathcal{N}}_X, \tilde{\mathcal{N}}_Y$) via rank-revealing factorizations.

   
Backed by an unconditional poly-logarithmic stability bound on the perturbed Lebesgue constant ($\tilde{\Lambda}_p \le \mathcal{O}(p^C \log p)$), LoCCA prevents Runge-type boundary instabilities, achieves up to a **$\sim 10\times$ speedup over ACA**, retains wall-clock times invariant to rank and tolerance, and maintains strict error control on arbitrary geometries where classical SI fails.

---

## 📂 Repository Structure

* 📂 `src/` — Core reusable codebase.
* 📂 `test_run/` — Standard, clean examples demonstrating how to execute.
* 📂 `tests/` — Benchmark experiment suites.
* 📂 `OUTPUTS/` — Designated storage for benchmark CSV logs and generated figures.

---

## 🛠️ Getting Started Locally

Follow these steps to clone, configure, and execute this codebase on your local machine.

### 1. Clone the Repository
Open your terminal and clone this repository into your desired directory:

```bash
git clone https://github.com/SAFRAN-LAB/LoCCA.git
cd LoCCA
```

### 2. Environment Setup & Dependencies
This project is fully reproducible using Julia's built-in package manager, which uses the `Project.toml` and `Manifest.toml` files in the root directory.

#### Required Packages
* `LinearAlgebra` & `Printf` (Standard Libraries)
* `LowRankApprox` (For Rank-Revealing QR factorizations and baseline comparisons)
* `NearestNeighbors` (For fast spatial $k$-d tree searches)
* `DataFrames` & `CSV` (For benchmark metrics tracking and CSV exporting)
* `Plots` (For error decay curves and visualization)


#### Automatic Installation
To instantiate the exact environment with correct dependency versions, run the following command directly from your terminal inside the project directory:
```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
Alternatively, you can open the standard Julia REPL, enter the package manager prompt by typing `]`, and execute:
```
pkg> activate .
pkg> instantiate
```





