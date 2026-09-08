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

## 🚀 Usage
### Running the Primary Test Example
To verify your installation and setup, execute the baseline example from your terminal:
```
julia --project=. test_run/test_LoCCA.jl
```

Upon execution, the script verifies package instantiation, displays configuration parameters, reports relative approximation error, and prints the confirmation status:




```text
  Activating project at `~/Desktop/LoCCA`

============================================================
                    EXPERIMENT DETAILS
============================================================
Kernel Choice         : kernel_1
Dimension             : 2
Grid Size             : 65^2
Number of Test        : 2
Chebyshev Grid Size   : 7^2
Nbd Size l            : 1
Target accuracy       : 1.0e-9
Target Hypercube      : [-3.0 -1.0; 0.0 2.0]
Source Hypercube      : [1.0 3.0; 0.0 2.0]
============================================================

LoCCA Error : 2.430787e-09
Obtained rank is r = 28
------------------------------------------------------------
 [PASS] Quick test completed successfully.
        • Core modules and dependencies are verified.
        • You can now run the other experiment scripts.
```





## ⚙️ Configuration & Parameter Tuning

Simulation parameters, kernel selections, and geometric bounds can be modified without altering the core solver routines. Update the parameter module located at: 📂 `test_run/config.jl` (or inside each specific experiment folder under `tests/`).

### Key Parameters Available for Tuning:

* **`kernel_choice`**: Select the kernel function to test. Supported kernel choices include:
  * `"kernel_1"`: Coulomb / Laplace potential ` "1/r" `
  * `"kernel_2"`: 2D Logarithmic potential `"log r"`
  * `"kernel_3"`: Helmholtz oscillatory kernel `" cos r/r"`
  * `"kernel_4"`: Gaussian RBF `"exp(-r^2)"`
  * `"kernel_5"`: Multiquadric RBF `"√(1 + r^2)"`
* **`chev_nodes_p` ($p$)**: Univariate Chebyshev degree per coordinate axis (yielding $P = p^d$ total grid anchors across dimension $d$).
* **`nbd_size` ($l$)**: Cardinality of the local nearest-neighbor cluster selected around each Chebyshev anchor (default is $l = 1$).
* **`tol` ($\varepsilon$)**: Target precision threshold for low-rank skeleton recompression via rank-revealing QR.
* **`num_domain_nodes` ($N$)**: Physical point cloud resolution per coordinate dimension (yielding total points $N^d$ per domain).
* **`num_test`**: Number of independent Monte Carlo trials or random grid realizations.
* **`target_hypercube` & `source_hypercube`**: Bounding coordinate matrices $[d \times 2]$ defining target domain $\mathcal{X}$ and source domain $\mathcal{Y}$ (e.g., `[-3.0 -1.0; 0.0 2.0]` and `[1.0 3.0; 0.0 2.0]`).

---

### Troubleshooting Parameter & Convergence Warnings

If the verification script, while running the primary test example, finishes with the following output:

```text
 [WARN] LoCCA ran to completion, but relative error was above expected tolerance.
        Check your parameters in config.jl before starting full runs.
```


This warning indicates that the relative approximation error exceeded the prescribed threshold by more than an order of magnitude (`LoCCA_error > 10 * tol`). This is caused by parameter mismatches rather than installation issues.

**Common Causes & Fixes in `test_run/config.jl`:**

* **Chebyshev Order ($p$) is too low for the requested tolerance ($\varepsilon$):**
  * *Reason:* If `tol = 1e-9`, a univariate order like `p = 3` cannot resolve the analytical kernel interaction, causing the approximation to stall before reaching the target precision.
  * *Fix:* Increase `chev_nodes_p` (e.g., set `chev_nodes_p = 7` or `9`).
* **Domain Admissibility & Proximity (Far-Field vs. Vertex-Sharing Configurations):**
  * *Setup in `test_run/config.jl`:* The baseline verification script is preconfigured for **well-separated, far-field domains** ($\mathcal{X} = [-3,-1]\times[0,2]$ and $\mathcal{Y} = [1,3]\times[0,2]$). The default univariate Chebyshev order $p = 7$ is sufficient to reach the target precision $\varepsilon = 10^{-9}$.
  * *Fix:* If you alter the domain bounds in `config.jl` to test near-field or vertex-sharing domains, you must increase the univariate Chebyshev resolution (e.g., set `chev_nodes_p = 15` or `p = 45` as used in the paper's vertex-sharing experiments) to prevent the analytic truncation error from dominating the user tolerance $\varepsilon$.

---

### Running Extended Benchmark Experiments

After verifying the setup with `test_run/test_LoCCA.jl`, you can reproduce all numerical benchmarks, parameter sweeps, and stability tests from the paper using the experiment scripts under `tests/`:

#### 1. Multi-Kernel Accuracy Comparisons on Rectangular Domains
Evaluates LoCCA, ACA, and SI across five standard kernels on both uniform and unstructured point clouds, generating empirical error distributions and boxplots:
```bash
julia --project=. tests/01_rectangular_domains/error_comparision.jl
```

#### 2. Complex & Non-Convex Geometries (SI Breakdown Demonstrations)
Executes cross-approximation tests on disconnected and non-box geometries (such as concentric ring/annular domains and crescent-and-core setups) where rigid tensor-product SI suffers from sample starvation, while LoCCA preserves error control:
```
julia --project=. tests/02_complex_domains/concentric_multiple_test.jl
```
```
julia --project=. tests/02_complex_domains/arc_dot_multiple_test.jl
```

#### 3. Parameter Sensitivity Sweeps ($l$, $p$, $N$, $\varepsilon$)
Tracks algorithmic sensitivity, showing that accuracy is robust to neighborhood size $l$, scales predictably with Chebyshev resolution $p$ and point cloud size $N$, and maintains flat wall-clock execution times
```
julia --project=. tests/03_parameter_sensitivity/tol_vs_error.jl
```
```
julia --project=. tests/03_parameter_sensitivity/time_taken_grid_size.jl
```

#### 4. 1D Perturbed Chebyshev Stability (Mitigation of Runge's Phenomenon)
Validates the theoretical stability bound on the perturbed Lebesgue constant ($\tilde{\Lambda}_p \le \mathcal{O}(p^C \log p)$) against classical Runge-type boundary blow-up on $[-1, 1]$:
```
julia --project=. tests/04_runge_stability/random_perturbation_test.jl
```

#### 5. Localized Chebyshev-Accelerated Truncated SVD (LoC-TSVD)
Constructs orthogonal rank-$r$ approximations via localized Chebyshev CUR proxy sampling coupled with an economy-sized QR recompression step:
```
julia --project=. tests/05_LoC-TSVD/test_LoC-TSVD.jl
```

---

## 📊 Benchmark Results & Observations

Extensive numerical experiments demonstrate the computational efficiency and stability of LoCCA compared to classical approaches:

* **Speedup over ACA:** LoCCA achieves up to a **$\sim 10\times$ speedup** over Adaptive Cross Approximation (ACA) at equivalent relative accuracy. Its wall-clock execution time remains flat and invariant with respect to target rank $r$ and prescribed tolerance $\varepsilon$, avoiding the iterative global vector searches that cause ACA runtimes to climb.
* **Stability vs. Skeletonized Interpolation (SI):** Box-plot distributions confirm that LoCCA achieves compact error spreads comparable to SI on standard domains while eliminating the outlier spikes observed in ACA.
* **Robustness on Complex Domains:** On non-convex or disconnected geometries (e.g., crescent-and-core and concentric annular domains), classical SI suffers catastrophic breakdown ($\mathcal{O}(1)$ error) due to sample starvation on rigid Cartesian grids. LoCCA dynamically maps continuous anchors strictly to existing physical coordinates, maintaining uninterrupted error control.

---

## 👥 Authors & Acknowledgments

* **Sumit Singh** — Department of Mathematics & Wadhwani School of Data Science and AI, IIT Madras ([Email](mailto:sumit1315singh@gmail.com) / [ORCID](https://orcid.org/0009-0002-5581-5349))
* **Shrirup Dutta** — Department of Mathematics, IIT Madras ([Email](mailto:shrirupdutta@gmail.com) / [ORCID](https://orcid.org/0009-0005-3479-8895))
* **Sivaram Ambikasaran** — Department of Mathematics & Department of Data Science and AI, IIT Madras ([Email](mailto:sivaambi@dsai.iitm.ac.in) / [ORCID](https://orcid.org/0000-0003-2978-6281))

Developed at the **SAFRAN Research Lab**, Indian Institute of Technology Madras, Chennai, India.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.


---

## 📜 Citation

If you find this codebase, algorithmic framework, or theoretical analysis helpful in your research, please cite our manuscript:

```bibtex
@article{singh2026locca,
  title   = {LoCCA: Localized Chebyshev Cross Approximation for Kernel Matrix Factorization via Nodal Perturbation Stability},
  author  = {Singh, Sumit and Dutta, Shrirup and Ambikasaran, Sivaram},
  journal = {arXiv preprint},
  year    = {2026}
}
```





















