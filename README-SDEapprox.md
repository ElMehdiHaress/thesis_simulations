Create singularSDEs-approx.py
README

Overview

This Python script simulates sample paths of fractional Brownian Motion (fBm) using the Davies Harte method and approximates solutions to Stochastic Differential Equations (SDEs) via the Euler scheme. The script also includes functionality to estimate the error of the SDE approximation and visualize the results.

Dependencies

The script requires the following Python libraries:
- `matplotlib`
- `numpy`
- `scipy`
- `sklearn`

You can install these dependencies using pip:
     ```bash
     pip install matplotlib numpy scipy scikit-learn
     ```

Usage

To use the `run_simulation` function, you need to provide the following arguments:

- `true_size`: The time-step for the true solution.
- `MC`: The Monte-Carlo size.
- `H`: The Hurst parameter.
- `size_list`: A list of time steps.
- `nature`: The nature of the drift (e.g., 'void', 'reg', 'ind', 'dirac').
- `scaling`: The scaling parameter.
- `dim`: The dimension.
- `MC2`: The number of Monte-Carlo simulations for error estimation.

The function will generate logarithmic plots of the error with error bars, which help visualize the accuracy and variability of the simulation results.
