from matplotlib import pyplot as plt 
import numpy as np
from numpy import linalg as LA
from math import *
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def davies_harte(T, N, H):
    '''
    Generates sample paths of fractional Brownian Motion using the Davies Harte method
    
    args:
        T:      length of time
        N:      number of time steps within timeframe
        H:      Hurst parameter
    '''
    gamma = lambda k,H: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))  
    g = [gamma(k,H) for k in range(0,N)];    r = g + [0] + g[::-1][0:N-1]

    # Step 1 (eigenvalues)
    j = np.arange(0,2*N);   k = 2*N-1
    lk = np.fft.fft(r*np.exp(2*np.pi*complex(0,1)*k*j*(1/(2*N))))[::-1]

    # Step 2 (get random variables)
    Vj = np.zeros((2*N,2), dtype=complex); 
    Vj[0,0] = np.random.standard_normal();  Vj[N,0] = np.random.standard_normal()
    
    for i in range(1,N):
        Vj1 = np.random.standard_normal();    Vj2 = np.random.standard_normal()
        Vj[i][0] = Vj1; Vj[i][1] = Vj2; Vj[2*N-i][0] = Vj1;    Vj[2*N-i][1] = Vj2
    
    # Step 3 (compute Z)
    wk = np.zeros(2*N, dtype=complex)   
    wk[0] = np.sqrt((lk[0]/(2*N)))*Vj[0][0];          
    wk[1:N] = np.sqrt(lk[1:N]/(4*N))*((Vj[1:N].T[0]) + (complex(0,1)*Vj[1:N].T[1]))       
    wk[N] = np.sqrt((lk[0]/(2*N)))*Vj[N][0]       
    wk[N+1:2*N] = np.sqrt(lk[N+1:2*N]/(4*N))*(np.flip(Vj[1:N].T[0]) - (complex(0,1)*np.flip(Vj[1:N].T[1])))
    
    Z = np.fft.fft(wk);     fGn = Z[0:N] 
    fBm = np.cumsum(fGn)*(N**(-H))
    fBm = (T**H)*(fBm)
    path = np.array([0] + list(fBm))
    return path

def fBm(MC,N,H): #modifiy accordingly if dim > 1
    '''
    Generates many samples of paths of fractional Brownian Motion using the Davies Harte method
    
    args:
        MC:     number of samples
        N:      number of time steps within timeframe
        H:      Hurst parameter
    '''
    B = []
    for i in range(MC):
        B += [davies_harte(1, N, H)]
        #print('%Nb MC = '+str(i))
    B = np.real(np.array(B))
    return B


def drift(x,m, nature, dim):
    '''
    Generates an approximation of the drift evaluated on the array x
    
    args:
        x:      array
        m:      order of the approximation (high m = better approximation)
        nature: type of drift (dirac, indicator function,...)
        dim:    dimension
    '''
    if nature == 'void':
        return 0
    if nature == 'reg':
        return -x
    if nature == 'ind':
        return norm.cdf(x,0,1/m)
    if nature == 'dirac':
        return sqrt(m/(2*pi)) * np.exp(-(x**2) * m /2)

    
def Sde(size, true_size, approx, B, nature, dim, MC=1):
    '''
    Simulates the SDE via a Euler scheme
    
    args:
        time_step
        true_step: step used for the true solution
        approx:    order of approximation of the drift
        B:         fBm
        nature:    nature of the drift
        dim:       dimension
        MC:        Monte-Carlo size
    '''
    time_step = 1/size
    step = int(true_size/size)
    # Start at the singularity to see unexpected/nondeterministic trajectories
    s = np.zeros((MC,size+1)) #=np.zeros((MC,size+1,dim)) if higher dimension
    for i in range(0,size):
        s[:,i+1] = s[:,i] + time_step*drift(s[:,i],approx,nature,dim) + (B[:,(i+1)*step]-B[:,i*step])
    return s

def error(true_size, size_list, nature, scaling, H, dim, MC):
    inf_h = 1/true_size
    #DH = davies_harte(1,N,H)
    B = fBm(MC,true_size,H) #modifiy if dim>1
    #print('%Simulation of fBm done')

    inf_m = true_size**(scaling)
    #Simulation of the true SDE
    true_sde = Sde(true_size,true_size,inf_m,B,nature,dim, MC)
    #print('%Simulation of true/reference solution done')    

    error=[]

    for size in size_list:
        h = 1/size
        m = size**(scaling)
        index = np.linspace(0,true_size,size+1)
        index = [ int(index[i]) for i in range(size+1)] 
        true_sde_interpol = true_sde[:, index]
        #Simulation of SDE with time step h
        h_sde = Sde(size, true_size, m, B, nature, dim, MC)
        Ech = (h_sde-true_sde_interpol)
        S = np.mean(Ech, axis=0)
        incr = [(S[i+1]-S[i])**2 for i in range(len(S)-1)] #modify accordignly if dim >1 
        holder = np.array(incr)*size
        error.append(np.max(holder))

    #print('err=[err;',error,'];')

    return np.array(error)



def run_simulation(true_size, MC, H, size_list, nature, scaling, dim, MC2):
    '''
    Runs the SDE simulation and plots the results
        
    args:
            true_size: time-step for true solution
            MC:        Monte-Carlo size
            H:         Hurst parameter
            size_list: list of time steps
            nature:    nature of the drift
            scaling:   scaling parameter
            dim:       dimension
            MC2:       number of Monte-Carlo simulations for error estimation
    '''
    print('%MC=', MC, ' H=', H, ' h=', size_list, '/', true_size, ' for drift=', nature)
        

    # Running multiple examples for error estimation
    e = np.zeros((MC2, len(size_list)))
    print(e.shape)
    for i in range(MC2):
        err = error(true_size, size_list, nature, scaling, H, dim, MC)
        print(err.shape)
        e[i, :] = err
        print('Iteration {i+1}/{MC2} completed')
        
    res = np.mean(e, axis=0)
    err = np.sqrt(np.var(e, axis=0))  
    plt.loglog(size_list, res)
    plt.errorbar(size_list, res, yerr=err)
    plt.show()
    reg = LinearRegression().fit(np.log(np.array(size_list)).reshape(-1, 1), np.log(np.array(res)).reshape(-1, 1))
    print('rate of convergence is: '-reg.coef_[0] / 2)

# Example usage
  #true_size = 1280000
  #MC = 1
  #H = 0.3
  #size_list = [1250, 2500, 5000, 10000, 20000, 40000, 80000, 160000, 320000, 640000]
  #nature = 'dirac'
  #scaling = 0.5
  #dim = 1
  #MC2 = 1
  #run_simulation(true_size, MC, H, size_list, 'dirac', scaling, dim, MC2)
