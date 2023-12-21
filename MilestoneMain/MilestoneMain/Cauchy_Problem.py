
from numpy import  zeros, float64

def Cauchy_Problem (time_domain, temporal_scheme, F, U0, NL = None ):
    
    N_t = len(time_domain) - 1 
    N_ve = len(U0)
    U = zeros ((N_t + 1 , N_ve), dtype=float64)
    U[0, :]= U0
    
    for n in range (0,N_t):
        
        if NL != None:
            
            U[n + 1, :] = temporal_scheme(U[n, :], time_domain[n], time_domain[n + 1], F, NL)
           
        else:
            
            U[n + 1, :] = temporal_scheme(U[n, :], time_domain[n], time_domain[n + 1], F)
            
    return U
  
def Cauchy_problem_RK4_emb(time_domain, temporal_scheme, F, U0): 
    N, Nv = len(time_domain) - 1, len(U0)
    U = zeros((N + 1, Nv), dtype=float64)
    h = zeros(N + 1)

    U[0, :] = U0
    for n in range(N):
        U[n + 1, :], h[n] = temporal_scheme(U[n, :], time_domain[n], time_domain[n + 1], F)

    return U, h
