
from scipy.optimize import newton
from numpy import zeros, array, linspace, dot, float64,  eye
import numpy as np
from numpy.linalg import norm
from scipy.linalg import eig

import numpy as np

def Euler(U, t1, t2, F):
    
    return U + (t2 - t1) * F(U, t1)

def Crank_Nicolson(U, t1, t2, F):
    
    def ResidualCN(X):
        
        return X - U - (t2 -t1)/2 * (F(U, t1) + F(U, t2)) - (t2 -t1)/2 * (F(X, t1) + F(X, t2 + (t2 - t1)))
    
    return newton(ResidualCN, U)

def Inverse_Euler(U, t1, t2, F):

    def ResidualIE(G):
        
        return G - U - (t2 - t1) *  F(U, t2)

    return newton(func = ResidualIE, x0 = U)

def RK4(U, t1, t2, F):
    
    k1 = F(U,t2)
    k2 = F(U + (t2 - t1) * k1/2, t2 + (t2 -t1)/2)
    k3 = F(U + (t2 - t1) * k2/2, t2 + (t2 -t1)/2)
    k4 = F(U + (t2 - t1) * k3, t2 + (t2 - t1))
    
    return U + (t2 - t1) * (k1 + 2*k2 + 2*k3 + k4)/6


Tolerance = 1e-8
NL_fixed = 3
N_GBS_effort = 0

def GBS_Scheme(U1, t1, t2, F):
    global Tolerance
    global NL_fixed
    global N_GBS_effort

    if NL_fixed > 0:
        GBS_solution_NL(U1, t1, t2, F, NL_fixed)
    else:
        raise ValueError("NL_fixed must be greater than 0 for fixed scheme")

def GBS_solution_NL(U1, t1, t2, F, NL):
    N = mesh_refinement(NL)
    U = np.zeros((NL, len(U1)))

    for i in range(NL):
        Modified_midpoint_scheme(U1, t1, t2, F, U[i, :], N[i])

    U2 = Corrected_solution_Richardson(N, U)
    return U2

def mesh_refinement(Levels):
    return np.arange(1, Levels + 1)

def Modified_midpoint_scheme(U1, t1, t2, F, U2, N):
    h = (t2 - t1) / (2 * N)
    U = np.zeros((len(U1), 2 * N + 2))
    U[:, 0] = U1
    U[:, 1] = U[:, 0] + h * F(U[:, 0], t1)

    for i in range(1, 2 * N + 1):
        ti = t1 + i * h
        U[:, i + 1] = U[:, i - 1] + 2 * h * F(U[:, i], ti)

    U2[:] = (U[:, 2 * N + 1] + 2 * U[:, 2 * N] + U[:, 2 * N - 1]) / 4.0

def Corrected_solution_Richardson(N, U):
    h = 1.0 / (2 * N)
    x = h ** 2
    W = 1.0 / (x * (x - 1))
    Uc = np.dot(W / x, U) / np.sum(W / x)
    return Uc




#Funcion del Esquema Temporal Runge-Kutta Embebido

def adaptive_RK_emb(U, dt, t, F):
    # Set tolerance for error estimation
    tol = 1e-9
    # Obtain coefficients and orders for the Butcher array
    orders, Ns, a, b, bs, c = ButcherArray()
    # Estimate state at two different orders
    est1 = perform_RK(1, U, t, dt, F) 
    est2 = perform_RK(2, U, t, dt, F) 
    # Calculate optimal step size
    h = min(dt, calculate_step_size(est1 - est2, tol, dt, min(orders)))
    N_n = int(dt / h) + 1
    n_dt = dt / N_n
    est1 = U
    est2 = U

    # Perform multiple steps with the adaptive step size
    for i in range(N_n):
        time = t + i * dt / int(N_n)
        est1 = est2
        est2 = perform_RK(1, est1, time, n_dt, F)

    final_state = est2
    ierr = 0

    return final_state,h

# Function to perform one step of Runge-Kutta integration
def perform_RK(order, U1, t, dt, F):
    # Obtain coefficients and orders for the Butcher array
    orders, Ns, a, b, bs, c = ButcherArray()
    k = zeros([Ns, len(U1)])
    k[0, :] = F(U1, t + c[0] * dt)

    if order == 1: 
        for i in range(1, Ns):
            Up = U1
            for j in range(i):
                Up = Up + dt * a[i, j] * k[j, :]
            k[i, :] = F(Up, t + c[i] * dt)
        U2 = U1 + dt * matmul(b, k)

    elif order == 2:
        for i in range(1, Ns):
            Up = U1
            for j in range(i):
                Up = Up + dt * a[i, j] * k[j, :]
            k[i, :] = F(Up, t + c[i] * dt)
        U2 = U1 + dt * matmul(bs, k)

    return U2

# Function to calculate the optimal step size based on the estimated error
def calculate_step_size(dU, tol, dt, orders): 
    error = norm(dU)

    if error > tol:
        step_size = dt * (tol / error) ** (1 / (orders + 1))
    else:
        step_size = dt

    return step_size


# Function to define the Butcher array coefficients for a specific Runge-Kutta method
def ButcherArray(): 
    orders = [2, 1]
    Ns = 2 

    a = zeros([Ns, Ns - 1])
    b = zeros([Ns])
    bs = zeros([Ns])
    c = zeros([Ns])

    c = [0., 1.]
    a[0, :] = [0.]
    a[1, :] = [1.]
    b[:] = [1./2, 1./2]
    bs[:] = [1., 0.]

    return orders, Ns, a, b, bs, c

### Leap Frog ###
# def leap_frog(U, t1, t2, F):
#     dt = t2 - t1
#     N = len(U)
#     t_old = 0
    
#     if t1 < t_old or t_old == 0:
#         U2 = U + dt * F(U, t1)
#     else:
#         U2 = U0 + 2 * dt * F(U, t1)
#         U0 = U
    
#     t_old = t2

#     return U2
def leap_frog(U, t1, t2, F):
    dt = t2 - t1
    U_half = U + dt / 2 * F(U, t1)
    return U + dt * F(U_half, (t1 + t2) / 2)



def Embedded_RK( U, dt, t, F, q, Tolerance): 
    
    #(a, b, bs, c) = Butcher_array(q)
    #a, b, bs, c = Butcher_array(q)
 
    N_stages = { 2:2, 3:4, 8:13  }
    Ns = N_stages[q]
    a = zeros( (Ns, Ns), dtype=float64) 
    b = zeros(Ns); bs = zeros(Ns); c = zeros(Ns) 
   
    if Ns==2: 
     
     a[0,:] = [ 0, 0 ]
     a[1,:] = [ 1, 0 ] 
     b[:]  = [ 1/2, 1/2 ] 
     bs[:] = [ 1, 0 ] 
     c[:]  = [ 0, 1]  

    elif Ns==13: 
       c[:] = [ 0., 2./27, 1./9, 1./6, 5./12, 1./2, 5./6, 1./6, 2./3 , 1./3,   1., 0., 1.]

       a[0,:]  = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0] 
       a[1,:]  = [ 2./27, 0., 0., 0., 0., 0., 0.,  0., 0., 0., 0., 0., 0] 
       a[2,:]  = [ 1./36 , 1./12, 0., 0., 0., 0., 0.,  0.,0., 0., 0., 0., 0] 
       a[3,:]  = [ 1./24 , 0., 1./8 , 0., 0., 0., 0., 0., 0., 0., 0., 0., 0] 
       a[4,:]  = [ 5./12, 0., -25./16, 25./16., 0., 0., 0., 0., 0., 0., 0., 0., 0]
       a[5,:]  = [ 1./20, 0., 0., 1./4, 1./5, 0., 0.,0., 0., 0., 0., 0., 0] 
       a[6,:]  = [-25./108, 0., 0., 125./108, -65./27, 125./54, 0., 0., 0., 0., 0., 0., 0] 
       a[7,:]  = [ 31./300, 0., 0., 0., 61./225, -2./9, 13./900, 0., 0., 0., 0., 0., 0] 
       a[8,:]  = [ 2., 0., 0., -53./6, 704./45, -107./9, 67./90, 3., 0., 0., 0., 0., 0] 
       a[9,:]  = [-91./108, 0., 0., 23./108, -976./135, 311./54, -19./60, 17./6, -1./12, 0., 0., 0., 0] 
       a[10,:] = [ 2383./4100, 0., 0., -341./164, 4496./1025, -301./82, 2133./4100, 45./82, 45./164, 18./41, 0., 0., 0] 
       a[11,:] = [ 3./205, 0., 0., 0., 0., -6./41, -3./205, -3./41, 3./41, 6./41, 0., 0., 0]
       a[12,:] = [ -1777./4100, 0., 0., -341./164, 4496./1025, -289./82, 2193./4100, 51./82, 33./164, 19./41, 0.,  1., 0]
      
       b[:]  = [ 41./840, 0., 0., 0., 0., 34./105, 9./35, 9./35, 9./280, 9./280, 41./840, 0., 0.] 
       bs[:] = [ 0., 0., 0., 0., 0., 34./105, 9./35, 9./35, 9./280, 9./280, 0., 41./840, 41./840]     
     

    
    k = RK_stages( F, U, t, dt, a, c ) 
    Error = dot( b-bs, k )

    dt_min = min( dt, dt * ( Tolerance / norm(Error) ) **(1/q) )
    N = int( dt/dt_min  ) + 1
    h = dt / N
    Uh = U.copy()

    for i in range(0, N): 

        k = RK_stages( F, Uh, t + h*i, h, a, c ) 
        Uh += h * dot( b, k )

    return Uh


def RK_stages( F, U, t, dt, a, c ): 

     k = zeros( (len(c), len(U)), dtype=float64 )

     for i in range(len(c)): 

        for  j in range(len(c)-1): 
          Up = U + dt * dot( a[i, :], k)

        k[i, :] = F( Up, t + c[i] * dt ) 

     return k 

def adaptive_RK_emb(U, dt, t, F):
    # Set tolerance for error estimation
    tol = 1e-9
    # Obtain coefficients and orders for the Butcher array
    orders, Ns, a, b, bs, c = ButcherArray()
    # Estimate state at two different orders
    est1 = perform_RK(1, U, t, dt, F) 
    est2 = perform_RK(2, U, t, dt, F) 
    # Calculate optimal step size
    h = min(dt, calculate_step_size(est1 - est2, tol, dt, min(orders)))
    N_n = int(dt / h) + 1
    n_dt = dt / N_n
    est1 = U
    est2 = U

    # Perform multiple steps with the adaptive step size
    for i in range(N_n):
        time = t + i * dt / int(N_n)
        est1 = est2
        est2 = perform_RK(1, est1, time, n_dt, F)

    final_state = est2
    ierr = 0

    return final_state,h

# Function to perform one step of Runge-Kutta integration
def perform_RK(order, U1, t, dt, F):
    # Obtain coefficients and orders for the Butcher array
    orders, Ns, a, b, bs, c = ButcherArray()
    k = zeros([Ns, len(U1)])
    k[0, :] = F(U1, t + c[0] * dt)

    if order == 1: 
        for i in range(1, Ns):
            Up = U1
            for j in range(i):
                Up = Up + dt * a[i, j] * k[j, :]
            k[i, :] = F(Up, t + c[i] * dt)
        U2 = U1 + dt * (b @ k)

    elif order == 2:
        for i in range(1, Ns):
            Up = U1
            for j in range(i):
                Up = Up + dt * a[i, j] * k[j, :]
            k[i, :] = F(Up, t + c[i] * dt)
        U2 = U1 + dt * (b @ k)

    return U2

# Function to calculate the optimal step size based on the estimated error
def calculate_step_size(dU, tol, dt, orders): 
    error = norm(dU)

    if error > tol:
        step_size = dt * (tol / error) ** (1 / (orders + 1))
    else:
        step_size = dt

    return step_size


# Function to define the Butcher array coefficients for a specific Runge-Kutta method
def ButcherArray(): 
    orders = [2, 1]
    Ns = 2 

    a = zeros([Ns, Ns - 1])
    b = zeros([Ns])
    bs = zeros([Ns])
    c = zeros([Ns])

    c = [0., 1.]
    a[0, :] = [0.]
    a[1, :] = [1.]
    b[:] = [1./2, 1./2]
    bs[:] = [1., 0.]

    return orders, Ns, a, b, bs, c