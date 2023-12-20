from  numpy import array, zeros, reshape, float64

#Problema de Cauchy 
def Cauchy_problem( F, t, U0, Temporal_scheme): 

 N, Nv=  len(t)-1, len(U0)
 U = zeros( (N+1, Nv), dtype=float64) 

 U[0,:] = U0

 for n in range(N):

     U[n+1,:] = Temporal_scheme( U[n, :], t[n+1] - t[n], t[n],  F) 

 return U

#Problema de Cauchy para coger el dt del runge kutta embebido en cada t
def Cauchy_problem_RK4_emb( F, t, U0, Temporal_scheme): 

 N, Nv=  len(t)-1, len(U0)
 U = zeros( (N+1, Nv), dtype=float64) 

 U[0,:] = U0
 h = zeros(N+1)
 for n in range(N):

     U[n+1,:],h[n] = Temporal_scheme( U[n, :], t[n+1] - t[n], t[n],  F) 

 return U,h


#Problema de Cauchy para el Leap-Frog
def Cauchy_problem_LP(F, t, U0, temporal_scheme):
    dt = 0.01
    Nv = len(U0)  # number of rows needed
    N = len(t) - 1  # number of columns needed
    U = zeros((Nv, N + 1), dtype=float64)
    U[:, 0] = U0
    U[:,1] = U[:,0] + dt*F(U[:,0], t[0])

    for i in range(1,N):
        U[:, i + 1] = temporal_scheme(U[:, i], U[:, i-1], t[i], t[i+1]-t[i], F)
        
    return U


#Problema de Cauchy para el GBS
def Cauchy_Problem_GBS (time_domain, temporal_scheme, F, U0, NL = None ):
    
    N_t = len(time_domain) - 1 
    N_ve = len(U0)
    U = zeros ((N_t + 1 , N_ve), dtype=float64)
    U[0, :]= U0
    
    for n in range (0,N_t):
            
        U[n + 1, :] = temporal_scheme(U[n, :], time_domain[n], time_domain[n + 1], F, NL)
            
    return U