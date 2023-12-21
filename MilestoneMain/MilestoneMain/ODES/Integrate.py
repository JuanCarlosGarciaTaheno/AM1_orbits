from numpy import array, zeros, log10, ones, vstack  
from numpy.linalg import norm, lstsq
from time import process_time


def Integrate_ODE(U0, F, t, scheme):
    N = len(t)-1; Nv = len(U0)           # N = numero evaluaciones, Nv = Numero variables (x,y,vx,vy...)
    U = zeros( (Nv, N+1))
    U[:,0] = U0

    for i in range(N):
        U[:,i+1] = scheme(F, U[:,i], t[i+1] - t[i], t[i])
        
    return U

def Integrate_ODE_GBS(U0, F, t, scheme, NL_fixed):
    N = len(t)-1; Nv = len(U0)           # N = numero evaluaciones, Nv = Numero variables (x,y,vx,vy...)
    U = zeros( (N+1, Nv))
    U[0,:] = U0

    for i in range(N):
        U[i+1,:] = scheme( U[i,:], t[i], t[i+1], F, NL_fixed)
        
    return U
