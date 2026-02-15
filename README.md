# Battery Parameter Estimation: OLS vs. Weighted TLS

## 🔋Project Overview

This repository contains a MATLAB simulation framework designed to compare parameter estimation techniques for Lithium-Ion batteries. Specifically, it addresses the problem of estimating the 
**Internal Resistance ($R_0$)** curve as a function of **State of Charge (SoC)** when **both** variables are subject to measurement noise.

Standard regression methods (Ordinary Least Squares - OLS) assume the input variable ($x$, here SoC) is noise-free. In real-world BMS applications, SoC is an estimated value derived from noisy voltage sensors, making OLS statistically biased. This project implements and benchmarks **Total Least Squares (TLS)** algorithms to correct this bias.


### Key Features

* **Li-Ion Physics Simulation:** Generates synthetic discharge data with realistic OCV-SoC curves and variable internal resistance.

* **Noise Injection:** Adds Gaussian noise to Voltage and Current sensors to simulate real hardware.

* **TLS Iterative Solver:** Implements a linearized, iterative approach (Newton-Gauss style) for TLS.

* **TLS SVD Solver:** Implements the Singular Value Decomposition approach, demonstrating the importance of "Whitening" (Weighting) when noise variances on $x$ and $y$ differ.

* **Piecewise Linear Fitting:** Segments the nonlinear $R_0$ curve into linear blocks for estimation.



---



## 📂 Repository Structure

### 1. `dataset_generator.m` (The Simulator)

This script simulates a battery pulse-discharge test.

* **Physics:** Uses a 4th-order polynomial for OCV and $R_0$ dependence on SoC.

* **Procedure:** Alternates between High-Current Pulses and Rest periods.

* **Measurement Generation:**

* Calculates "True" $V$ and $I$.

* Injects Gaussian noise ($sigma_V$, $sigma_I$).

* **Derives Noisy SoC:** Uses the noisy Voltage reading during rest (OCV) to look up SoC.

* **Derives Noisy $R_0$:** Uses Ohm's law on current steps ($Delta V / Delta I$).

* **Output:** Saves `dataset.mat` in a `../res/` directory containing both clean (ground truth) and noisy data, along with calculated noise statistics ($sigma_{SoC}$, $sigma_{R0}$).



### 2. `TLS_iter_lin.m` (Iterative Method)

Solves the Errors-in-Variables problem using an iterative minimization of the weighted sum of squared residuals.

* **Methodology:**

1.  Segments data into SoC ranges (e.g., 0-28%, 28-78%, etc.).

2.  Initializes slope estimates using OLS.

3.  Iteratively updates the slope ($a$) and intercept ($b$) using a Jacobian-based approach weighted by the noise covariance matrix.

* **Visualization:** Plots the convergence of the slope from the OLS guess to the final TLS solution over $N$ iterations.



### 3. `TLS_svd.m` (Algebraic Method)

Solves the problem using Singular Value Decomposition (Orthogonal Regression).

* **The "Whitening" Concept:**

* **Unweighted SVD:** Assumes noise on SoC and $R_0$ is equal (isotropic). This usually yields incorrect results in battery applications where units differ ($V$ vs $Omega$).

* **Weighted (Whitened) SVD:** Scales the data by the inverse of their standard deviations ($1/\sigma_x, 1/\sigma_y$) before performing SVD. This minimizes the Mahalanobis distance rather than the Euclidean distance.

* **Comparison:** Benchmarks Unweighted vs. Weighted SVD against the True parameter values.



---



## 🗒️ Notes

### Prerequisites

* MATLAB (R2021b or newer recommended).

* Statistics and Machine Learning Toolbox (for `polyfit`/`interp1` functions).

### Installation & Run Order

1.  Clone the repository.

2.  Ensure you have a directory named `res` at the parent level or adjust the filepath in the scripts (`../res/dataset.mat`).

3.  **Step 1:** Run the generator to create the data.

    ```matlab

    dataset_generator

    ```

4.  **Step 2:** Run the solvers to analyze the data.

    ```matlab

    TLS_iter_lin  % For the iterative approach

    TLS_svd       % For the SVD approach

    ```



---



## 📊 Methodology: Why TLS?

In a standard linear model $y = ax + b$:

1.  **OLS (Ordinary Least Squares):** Minimizes the vertical distance (green lines). Assumes $x$ is perfect.

2.  **TLS (Total Least Squares):** Minimizes the orthogonal distance (perpendicular) to the line. Assumes noise on both $x$ and $y$.

**The Battery Context:**

* $y$ = Calculated Resistance ($R_{meas}$). Noise comes from $V/I$ sensors.

* $x$ = Measured SoC ($SoC_{meas}$). Noise comes from OCV lookups via $V$ sensors.

Since $\sigma_{SoC}$, OLS suffers from **Attenuation Bias** (the slope is underestimated towards zero). This project demonstrates that **Weighted TLS** recovers the true physical parameters of the battery much more accurately than OLS.
