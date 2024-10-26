# thesis_simulations
Here I gather the code for the simulations presented in my thesis. This includes:

1) Numerical simulations of fractional Stochastic Differential Equations with distributional drift.
   - See singularSDEs-approx.py for the code. It contains:  
      - Simulation of the fractional Brownian motion
      - Implementation of a tamed-Euler scheme
      - Error plots

3) Numerical simulations of the Stochastic Heat Equation with a Dirac reaction term.
   - See singularSHE-visual.py for the code. It contains:
      - Implementation of a tamed-Euler and finite-difference scheme
      - Plots of the evolution in time of the solution

5) Estimation of parameters in the fractional Ornstein-Uhlenbeck model.
   - See estimationSDEs.ipynb for the code. It contains:
      - Simulation of the fractional Brownian motion
      - Simulation of the fractional Onstein-Uhlenbeck process
      - Implementation of the estimators
      - Histograms with highlighted mean and standard deviation of the estimators
  

