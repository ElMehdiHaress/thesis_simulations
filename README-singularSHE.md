"""
README for Stochastic Heat Equation Simulation

This project simulates and visualizes the stochastic heat equation using a finite-differences method in space and a tamed Euler scheme in time. The simulation includes options for different types of drift terms and noise intensities.

## Requirements

- Python 3.x
- numpy
- matplotlib
- tqdm
- scipy
- scikit-learn

You can install the required packages using pip:
    pip install numpy matplotlib tqdm scipy scikit-learn
    
Usage

To run the simulation, use the `run_simulation_she` function. Below is an explanation of the parameters and the output:

### Parameters

- `T` (int): Time limit for the simulation.
-   - `L` (int): Space limit for the simulation.
- `Nx` (int): Number of space points.
- `MC` (int): Number of Monte Carlo simulations.
- `nature` (str): Type of drift term. Options include 'dirac', 'ind', and 'bessel'. Feel free to add more drift examples in the function drift.
- `scaling` (float): Scaling factor for the drift term.
- `Fps` (int): Frames per second for the animation.
- `c` (float): Constant for the CFL condition.
- `dW_intensity` (float): Intensity of the noise term.

