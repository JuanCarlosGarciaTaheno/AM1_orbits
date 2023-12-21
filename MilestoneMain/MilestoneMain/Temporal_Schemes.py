from scipy.optimize import newton
from ERK_functions import butcher, RK_stages, StepSize
from Newton import Jacobian2 
from numpy.linalg import eig, norm
from numpy import zeros,eye, array, zeros_like, prod, where, dot
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
def Euler(U, t1, t2, F):
    
    return U + (t2 - t1) * F(U, t2)


def Crank_Nicolson(U, t1, t2, F):
    
    def ResidualCN(X):
        
        return X - a - (t2 -t1)/2 * F(X, t2 + (t2 + t1))
    
    a = U + (t2 - t1)/2 * F(U, t2)
    
    return newton(ResidualCN, U, maxiter = 600)


def Inverse_Euler(U, t1, t2, F):

    def ResidualIE(G):
        
        return G - U - (t2 - t1) *  F(G, t2)

    return newton(func = ResidualIE, x0 = U)


def RK4(U, t1, t2, F):
    
    k1 = F(U,t2)
    k2 = F(U + (t2 - t1) * k1/2, t2 + (t2 -t1)/2)
    k3 = F(U + (t2 - t1) * k2/2, t2 + (t2 -t1)/2)
    k4 = F(U + (t2 - t1) * k3, t2 + (t2 - t1))
    
    return U + (t2 - t1) * (k1 + 2*k2 + 2*k3 + k4)/6
    
def Leap_Frog(U, t1, t2, F):
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


def ERK(U, t1, t2, F):

    tol = 1e-9
    
    # Calculates Runge-Kutta of order 1 and order 2 
    stage1 = RK_stages(1, U, t2, (t2 - t1), F)  
    stage2 = RK_stages(2, U, t2, (t2 - t1), F) 
    
    # Define the butcher array
    orders, Ns, a, b, bs, c = butcher()
    
    # Determine the minimum step size between dt and the stepsize, which compares the error with the tolerance
    h = min((t2 - t1), StepSize(stage1 - stage2, tol, (t2 - t1),  min(orders)))
    N_n = int((t2 - t1)/h) + 1        # Number of steps to update solution U2
    n_dt = (t2 - t1) / int(N_n)           
    stage1 = U
    stage2 = U

    for i in range(N_n):
        time = t2 + i * n_dt
        stage1 = stage2
        stage2 = RK_stages(1, stage1, time, n_dt, F)
        
    # Final solution
    U2 = stage2
    
    ierr = 0

    return U2
def ERK2(U, t1, t2, F):

    tol = 1e-9
    
    # Calculates Runge-Kutta of order 1 and order 2 
    stage1 = RK_stages(1, U, t2, (t2 - t1), F)  
    stage2 = RK_stages(2, U, t2, (t2 - t1), F) 
    
    # Define the butcher array
    orders, Ns, a, b, bs, c = butcher()
    
    # Determine the minimum step size between dt and the stepsize, which compares the error with the tolerance
    h = min((t2 - t1), StepSize(stage1 - stage2, tol, (t2 - t1),  min(orders)))
    N_n = int((t2 - t1)/h) + 1        # Number of steps to update solution U2
    n_dt = (t2 - t1) / int(N_n)           
    stage1 = U
    stage2 = U

    for i in range(N_n):
        time = t2 + i * n_dt
        stage1 = stage2
        stage2 = RK_stages(1, stage1, time, n_dt, F)
        
    # Final solution
    U2 = stage2
    
    ierr = 0

    return U2,h


def GBS_solution_NL(U1, t1, t2, F, NL):
    # Obtiene la secuencia de refinamiento de malla N
    N = mesh_refinement(NL)
    # Inicializa una matriz U para almacenar las soluciones en diferentes niveles
    U = zeros((NL, len(U1)))
    # Inicializa el resultado final U2
    U2 = zeros_like(U1)

    # Itera sobre los niveles de refinamiento
    for i in range(NL):
        # Aplica el esquema Modified Midpoint para avanzar en el tiempo y almacena la soluci�n en U
        Modified_midpoint_scheme(F, t1, t2, U1, U[i, :], N[i])
    
        # Aplica la correcci�n Richardson para obtener la soluci�n final
    U2 = Corrected_solution_Richardson(N, U)

    return U2

def mesh_refinement(Levels):
    N_Romberg = array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    N_Burlirsch = array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    N_Harmonic = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # N = N_Romberg[:Levels]
    # N = N_Burlirsch[:Levels]
    N = N_Harmonic[:Levels]

    return N

def Corrected_solution_Richardson(N, U):
    NL = len(N)
    Uc = zeros(U.shape[1])
    
    # Calcula los coeficientes de Lagrange para la correcci�n Richardson
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