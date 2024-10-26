import numpy as np
import matplotlib.pyplot as plt
from math import *
from tqdm import tqdm
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy import stats, interpolate
import scipy.integrate as integrate
from numpy.linalg import inv

# Function to create a finite-differences matrix for the 1D heat equation
def finite_differences_matrix(N):
    """
    Creates a finite-differences matrix for the 1D heat equation with
    Dirichlet boundary conditions.
        N : number of space points, int
    Returns: N-2 x N-2 matrix    
    """
    n = N - 2
    A = diags([np.ones(n)*(2), -np.ones(n-1), -np.ones(n-1)], [0, 1, -1])
    return -A

# Function to define the initial condition
def init_cond(x):
    """
    Initial condition
        x : array
    Returns : len(x)-array    
    """
    return 10 * np.abs(x-1) * np.abs(x-0.5) * np.abs(np.sin(2*np.pi*(x - x**3)))

# Function to define the drift term
def drift(x, nature, scaling, dx, intensity):
    """
    Drift function
        x : array
        nature : string
        scaling : float
        dx : float
    Returns: len(x)-array    
    """
    m = dx**(-scaling)
    if nature == 'dirac':
        return intensity * np.sqrt(m / (2 * np.pi)) * np.exp(-(x-1)**2 * m / 2)
    if nature == 'ind':
        return norm.cdf(x, 0, 1/m)
    if nature == 'bessel':
        return intensity / x**(0.5)
    return 0

# Function to generate the noise term
def dW_func(Nx, MC,c):
    Nt = int((1/c) * (Nx**2))
    dt = 1 / Nt
    return (c * dt)**(1/4) * np.random.normal(size=(Nx+1, Nt+1, MC))

# Function to solve the stochastic heat equation
def Spde(drift, nature, scaling, Nx, L, T, initial_cond, MC, noise, dW, intensity, c):
    """
    Computes a tamed Euler in time, finite-differences in space, scheme for the stochastic heat equation
        drift : func
        nature : string
        scaling : float
        Nx : number of space points, float
        L : space limit, int
        T : time limit, int
        initial_cond : initial condition, func
        MC : Monte-Carlo, int
        noise : bool
    Returns : Nx x Nt x MC array if noise = True and Nx x Nt array if noise = False, where Nt is the number of time points.
    """
    Nx = dW.shape[0] - 1
    Nt = int((1/c) * (Nx**2))
    Dt = dW.shape[1] - 1
    MC = dW.shape[2]
    dt = 1 / Nt
    dx = 1 / Nx

    x = np.linspace(0, 1, Nx+1)
    u0 = initial_cond
    ue = np.zeros((Nx+1, Dt+1, MC))
    ue_imp = np.zeros((Nx+1, Dt+1, MC))
    
    if u0[0].shape == ():
        ue[:, 0, :] = u0[:, np.newaxis]
        ue_imp[:, 0, :] = u0[:, np.newaxis]
    else:
        ue[:, 0, :] = u0
        ue_imp[:, 0, :] = u0
        
    diff_matrix = finite_differences_matrix(Nx+1)
    N = Nx - 1
    inv_diff = inv(np.diag(np.ones(N)) - c * diff_matrix)
    
    for n in tqdm(range(0, Nt)):
        if noise:
            #ue is for explicit and ue_imp is for implicit Euler
            ue[1:Nx, n+1, :] = dt * drift(ue[1:Nx, n, :], nature, scaling, dx, intensity) \
                               + ue[1:Nx, n, :] + c * (diff_matrix * ue[1:Nx, n, :]) \
                               + dW[1:Nx, n, :]
            ue_imp[1:Nx, n+1, :] = inv_diff * (dt * drift(ue_imp[1:Nx, n, :], nature, scaling, dx, intensity) \
                                               + ue_imp[1:Nx, n, :] \
                                               + dW[1:Nx, n, :])
        else:
            ue[1:Nx, n+1, 0] = dt * drift(ue[1:Nx, n, 0], nature, scaling, dx, intensity) \
                               + ue[1:Nx, n, 0] + c * (diff_matrix * ue[1:Nx, n, 0])
                                                
    if noise:
        return ue, ue_imp
    else:
        return ue[:, :, 0]

# Function to visualize the evolution of the stochastic heat equation in time
def Sheat_visual(T, L, Nx, reaction, nature, scaling, MC, Fps, dW, c):
    """
    Plots a visual of the stochastic heat equation with a dirac drift
        T : time limit, int
        L : space limit, int
        Nx : number of space points
        reaction : drift function, func
        nature : string
        scaling : float
        MC : Monte-Carlo, int
        Fps : int
    """
    dx = L / Nx
    dt = c * (Nx)**(-2)
    Nt = int(T / dt)
    
    x = np.linspace(0, L, Nx+1)
    u0 = init_cond(x)
    sample = int(np.random.rand() * MC)

    true_spde, true_spde_imp = Spde(reaction, nature, scaling, Nx, L, T, u0, MC, True, dW, 100, c) 
    #use either explicit or implicit scheme for visuals
    
    small_average = np.mean(true_spde_imp[:, :, 0:int(2*MC/8)], axis=2)
    small_average1 = np.mean(true_spde_imp[:, :, int(2*MC/8):int(4*MC/8)], axis=2)
    small_average2 = np.mean(true_spde_imp[:, :, int(4*MC/8):int(6*MC/8)], axis=2)
    small_average3 = np.mean(true_spde_imp[:, :, int(6*MC/8):MC], axis=2)
    
    fig, ax = plt.subplots()
    line_av_ = ax.plot(x, u0)
    y = -5 * np.ones(len(x))
    
    ax.set_title('Evolution of the Stochastic heat equation with a dirac drift')
    ax.legend(loc='upper left', frameon=False)
    ax.set_xlabel('x')
    ax.set_xlim(0, 1)
    ax.set_ylim(-5.1, 5.1)
    
    def animate(frame_num):
      #update frame
        ax.set_xlabel('x')
        ax.set_xlim(0, 1)
        ax.set_ylim(-5.1, 5.1)
        
        x = np.linspace(0, L, Nx+1)
        
        y_av = small_average[:, frame_num]
        y_av1 = small_average1[:, frame_num]
        y_av2 = small_average2[:, frame_num]
        y_av3 = small_average3[:, frame_num]
        
        line_av, = ax.plot(x, y_av, facecolor='royalblue', alpha=0.8)
        line_av1, = ax.plot(x, y_av1, facecolor='royalblue', alpha=0.7)
        line_av2, = ax.plot(x, y_av2, facecolor='royalblue', alpha=0.6)
        line_av3, = ax.plot(x, y_av3, facecolor='royalblue', alpha=0.5)

        ax.legend(loc='upper left', frameon=False)
        
        return line_av, line_av1, line_av2, line_av3
    
    anim = FuncAnimation(fig, animate, frames=range(Nt), interval=50, blit=True)
    plt.show()
    anim.save(filename="Sheat_average_visual_time2.mp4", dpi=80, fps=Fps)

def run_simulation_she(T, L, Nx, MC, nature, scaling, Fps, c, dW_intensity):
    """
    Runs the simulation for the stochastic heat equation and visualizes the result.
        T : time limit, int
        L : space limit, int
        Nx : number of space points
        MC : Monte-Carlo, int
        nature : string
        scaling : float
        Fps : int
        c : float, constant (CFL condition)
        dW_intensity : float, intensity of the noise term
    """
    dW = dW_intensity * dW_func(Nx, MC, c)
    Sheat_visual(T, L, Nx, drift, nature, scaling, MC, Fps, dW, c)

# Example usage
#T = 1
#L = 1
#Nx = 2**5
#MC = 10000
#nature = 'dirac'
#scaling = 0.5
#Fps = 10
#c = 0.4
#dW_intensity = 15

#run_simulation_she(T, L, Nx, MC, nature, scaling, Fps, c, dW_intensity)
