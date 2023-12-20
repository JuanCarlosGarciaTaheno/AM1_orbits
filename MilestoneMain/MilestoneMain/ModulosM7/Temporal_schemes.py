from scipy.optimize import newton
from numpy import zeros, dot, matmul, array, prod, zeros, zeros_like, dot, sum, where
from numpy.linalg import norm,eig
from numpy import matmul
#Funcion del Esquema Temporal de Euler
def Euler( U, dt, t, F): 

   return U + dt * F( U,t ) 

#Funcion del Esquema Temporal Runge-Kutta de orden 4
def RK4( U, dt, t, F): 

     k1 = F( U, t)
     k2 = F( U + dt * k1/2, t + dt/2 )
     k3 = F( U + dt * k2/2, t + dt/2 )
     k4 = F( U + dt * k3,   t + dt   )
 
     return U + dt * ( k1 + 2*k2 + 2*k3 + k4 )/6

#Funcion del Esquema Temporal de Euler Inverso
def Inverse_Euler( U, dt, t, F): 

   def  Residual_IE(X):  

          return  X - U - dt * F(X, dt + t) 

   return  newton( Residual_IE, U )
    
#Funcion del Esquema Temporal Implicito de Crank Nicolson
def Crank_Nicolson(U, dt, t, F): 

    def Residual_CN(X): 
         
         return  X - a - dt/2 *  F(X, t + dt)

    a = U  +  dt/2 * F( U, t)  
    return newton( Residual_CN, U )

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


# Funcion del esquema temporal Leap-Frog 
def Leapfrog(U2, U1, t, dt, F):
    return U1 + 2 * dt * F(U2, t)


# Funcion del esquema temporal Leap-Frog para el GBS
def leap_frog(U, t1, t2, F):
    dt = t2 - t1
    N = len(U)
    t_old = 0
    
    if t1 < t_old or t_old == 0:
        U2 = U + dt * F(U, t1)
    else:
        U2 = U0 + 2 * dt * F(U, t1)
        U0 = U
    
    t_old = t2

    return U2


#Funcion del esquema temporal GBS
def GBS(U1, t1, t2, F, NL):
    N = mesh_refinement(NL)
    U = zeros((NL, len(U1)))
    U2 = zeros_like(U1)

    for i in range(NL):
        Modified_midpoint_scheme(F, t1, t2, U1, U[i, :], N[i])

    U2 = Corrected_solution_Richardson(N, U)

    return U2

def mesh_refinement(Levels):
    N_Romberg = array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    N_Burlirsch = array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    N_Harmonic = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Uncomment one of the following lines based on the desired refinement sequence
    # N = N_Romberg[:Levels]
    # N = N_Burlirsch[:Levels]
    N = N_Harmonic[:Levels]

    return N

def Corrected_solution_Richardson(N, U):
    NL = len(N)
    Uc = zeros(U.shape[1])

    h = 1.0 / (2 * N)  # Leap Frog
    x = h**2  # even power of h (time step)

    if NL == 1:
        Lagrange = array([1.0])
        w = array([1.0])
    else:
        Lagrange = array([prod(where(x - x[j] == 0, 1, x / (x - x[j]))) for j in range(NL)])
        w = array([1 / prod(x[j] - x[x != x[j]]) for j in range(NL)])

    # Barycentric formula
    Uc = dot(w / x, U) / sum(w / x)

    # Lagrange formula (if needed)
    # Uc = np.dot(Lagrange, U)

    return Uc

def Modified_midpoint_scheme(F, t1, t2, U1, U2, N):
    h = (t2 - t1) / (2 * N)
    U = zeros((len(U1), 2 * N + 2))

    U[:, 0] = U1
    U[:, 1] = U[:, 0] + h * F(U[:, 0], t1)

    for i in range(1, 2 * N + 1):
        ti = t1 + i * h
        U[:, i + 1] = U[:, i - 1] + 2 * h * F(U[:, i], ti)

    # Valor promedio en t2 = t1 + 2 * N * h
    U2[:] = (U[:, 2 * N + 1] + 2 * U[:, 2 * N] + U[:, 2 * N - 1]) / 4.0