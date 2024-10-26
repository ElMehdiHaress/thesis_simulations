 Incrementing the Observations to Estimate all the Parameters in Fractional Additive SDEs

This Jupyter Notebook demonstrates the process of estimating parameters in fractional additive stochastic differential equations (SDEs) using discrete observations. The notebook is divided into several sections, each focusing on different aspects of the simulation and estimation process.

## Table of Contents

1. [Introduction](#introduction)
2. [Simulation of Fractional Brownian Motion](#simulation-of-fractional-brownian-motion)
3. [Simulation of Fractional Ornstein-Uhlenbeck Process](#simulation-of-fractional-ornstein-uhlenbeck-process)
4. [Extraction of the Sample](#extraction-of-the-sample)
5. [Minimization Procedure: Estimation of One Parameter](#minimization-procedure-estimation-of-one-parameter)
6. [Two Parameters Estimation](#two-parameters-estimation)
7. [Minimization Procedure: Estimation of Three Parameters](#minimization-procedure-estimation-of-three-parameters)

## Introduction

We consider the following one-dimensional Ornstein-Uhlenbeck process:

$$dY = -\xi_0 Y_s dt + \sigma_0 dB^{H_0}.$$

We have access to discrete observations of \(Y\) of the form \(\{Y_{kh}, k=1,2,..\}\) and we want to estimate the parameters \(\xi_0,\sigma_0\) and \(H_0\). The estimation procedure is based on minimizing the distance:

$$ \theta = (\xi,\sigma,H) \rightarrow d(\frac{1}{n} \sum_1^n \delta_{X_{kh}}, \frac{1}{N} \sum_1^{N} \delta_{L^{\theta}_{k\gamma}})$$

where \(d\) is a distance that is bounded by the \(p\)-Wassertein distance and 

$$X_. = (Y_., Y_{.+h}-Y_., Y_{.+2h}-Y_.,...,Y_{.+qh}-Y_)$$ 

is a vector that contains the solution \(Y\) and its increments. Similarly, 

$$L^\theta_.= (M^\theta_., M^\theta_{.+h}-M^\theta_., M^\theta_{.+2h}-M^\theta_{.+qh}-M^\theta_.)$$ 

where \(M^\theta\) is a simulated O-U process with the parameters \(\xi,\sigma\) and \(H\) (through a Euler scheme of step size \(\gamma\)).

## Simulation of Fractional Brownian Motion

We simulate the fractional Brownian motion using the Davies-Harte method. The function `davies_harte` generates sample paths of fractional Brownian Motion, and the function `fBm` generates multiple sample paths.

## Simulation of Fractional Ornstein-Uhlenbeck Process

We simulate the fractional Ornstein-Uhlenbeck process through a Euler scheme using the function `ornstein_uhlenbeck`.

## Extraction of the Sample

Given a set of parameters \(\theta\), we extract the associated sample \(X^\theta\) using the functions `increments`, `true_sample`, and `euler_sample`.

## Minimization Procedure: Estimation of One Parameter

We estimate one parameter at a time using gradient descent. The functions `loss_drift`, `loss_hurst`, `loss_diffusion`, and their corresponding oracle functions are used to compute the loss. The gradient descent is performed using the functions `descent_drift`, `descent_hurst`, and `descent_diffusion`.

## Two Parameters Estimation

We estimate two parameters simultaneously and compare the results between oracle and approximation. The functions `loss_drift_hurst`, `loss_hurst_sigma`, `loss_diffusion_drift`, and their corresponding oracle functions are used to compute the loss. The gradient descent is performed using the functions `descent_drift_hurst`, `descent_hurst_sigma`, and `descent_diffusion_drift`.

## Minimization Procedure: Estimation of Three Parameters

We estimate all three parameters simultaneously using gradient descent. The functions `loss_all`, `oracle_loss_all`, `grad_all`, and `oracle_grad_all` are used to compute the loss and gradients. The gradient descent is performed using the functions `descent_all` and `oracle_descent_all`.

## Examples

The notebook includes several examples demonstrating the usage of the functions and the estimation process. The examples are provided in the cells following each section.

## Variables

The following variables are used in this notebook:

- `B`: numpy.ndarray
- `B02`: numpy.ndarray
- `B04`: numpy.ndarray
- `B06`: numpy.ndarray
- `B08`: numpy.ndarray
- `H`: float
- `N`: int
- `OU1`: numpy.ndarray
- `acos`: builtin_function_or_method
- `acosh`: builtin_function_or_method
- `asin`: builtin_function_or_method
- `asinh`: builtin_function_or_method
- `atan`: builtin_function_or_method
- `atan2`: builtin_function_or_method
- `atanh`: builtin_function_or_method
- `ceil`: builtin_function_or_method
- `comb`: builtin_function_or_method
- `copysign`: builtin_function_or_method
- `cos`: builtin_function_or_method
- `cosh`: builtin_function_or_method
- `degrees`: builtin_function_or_method
- `dist`: builtin_function_or_method
- `e`: float
- `erf`: builtin_function_or_method
- `erfc`: builtin_function_or_method
- `exp`: builtin_function_or_method
- `expm1`: builtin_function_or_method
- `fabs`: builtin_function_or_method
- `factorial`: builtin_function_or_method
- `floor`: builtin_function_or_method
- `fmod`: builtin_function_or_method
- `frexp`: builtin_function_or_method
- `fsum`: builtin_function_or_method
- `gamma`: numpy.ufunc
- `gcd`: builtin_function_or_method
- `h`: float
- `hypot`: builtin_function_or_method
- `inf`: float
- `isclose`: builtin_function_or_method
- `isfinite`: builtin_function_or_method
- `isinf`: builtin_function_or_method
- `isnan`: builtin_function_or_method
- `isqrt`: builtin_function_or_method
- `lcm`: builtin_function_or_method
- `ldexp`: builtin_function_or_method
- `lgamma`: builtin_function_or_method
- `log`: builtin_function_or_method
- `log10`: builtin_function_or_method
- `log1p`: builtin_function_or_method
- `log2`: builtin_function_or_method
- `modf`: builtin_function_or_method
- `n`: int
- `nan`: float
- `nextafter`: builtin_function_or_method
- `p`: int
- `perm`: builtin_function_or_method
- `pi`: float
- `pow`: builtin_function_or_method
- `prod`: builtin_function_or_method
- `radians`: builtin_function_or_method
- `remainder`: builtin_function_or_method
- `sigma`: float
- `sin`: builtin_function_or_method
- `sinh`: builtin_function_or_method
- `sqrt`: builtin_function_or_method
- `t`: numpy.ndarray
- `tan`: builtin_function_or_method
- `tanh`: builtin_function_or_method
- `tau`: float
- `timestep`: float
- `trials`: int
- `trunc`: builtin_function_or_method
- `ulp`: builtin_function_or_method
- `xi`: int

## Conclusion

This notebook provides a comprehensive guide to simulating and estimating parameters in fractional additive SDEs. The methods and functions demonstrated here can be adapted and extended for various applications in stochastic processes and financial modeling.
```
