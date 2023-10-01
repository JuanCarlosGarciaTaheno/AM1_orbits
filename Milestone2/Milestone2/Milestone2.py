from numpy import array, zeros
import matplotlib.pyplot as plt 
from scipy.optimize import newton


# Milestone 2 : Prototypes to integrate orbits with functions.

        #Condiciones INICIALES

# Define condiciones iniciales

U0 = array([1.0, 0.0, 0.0, 1.0])  # Condiciones iniciales (x, y, vx, vy)
t0 = 0    # tiempo inicial
T = 10.0    # periodo o tiempo final
N = 1000    # numero de pasos
dt = 0.0001   # paso de tiempo

# Funcion para la fuerza de Kepler
def kepler_force(U, t):
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    r = (x**2 + y**2)**0.5
    return array([vx, vy, -x/(r**3), -y/(r**3)])

##############################################################################
# 1. Write a function called Euler to integrate one step. The function F(U, t)
# of the Cauchy problem should be input argument.


def Euler(U, dt, t, F): 

    return U + dt * F(U, t)

# los inputs parar la funcion Euler en un paso de Integracion N=1;
# tienen que ser el vector/matriz U
# el dt; paso de tiempo
# F la funcion del problema a integrar numericamente
# Por tanto; si el esquema es Euler y se busca el n+1, se busca la la definicion
# de como return Un+1= Un + dt*F(Un,t)

##############################################################################
# 2. Write a function called Crank_Nicolson to integrate one step.

#De la misma forma se busca la solucion en n+1 pero ahora el 
# esquema es implicito luego si el esquema es Un+1=Un+0.5*(F(Un+1,t)+F(Un,t))
# para resolverlo se busca una funcion tal que Un+1-Un-0.5*(F(Un+1,t)+F(Un,t)) sea lo pedido 
#y con ello se busque un 0 de la funcion

def Crank_Nicolson(U, dt, t, F ): 

    def Residual_CN(X): 
         
         return  X - a - dt/2 *  F(X, t + dt)

    a = U  +  dt/2 * F( U, t)  
    return newton( Residual_CN, U )
#Esta seria la correcion hecha en clase del CN

##############################################################################
# 3. Write a function called RK4 to integrate one step
def RK4(U, dt, t, F ): 

     k1 = F( U, t)
     k2 = F( U + dt * k1/2, t + dt/2 )
     k3 = F( U + dt * k2/2, t + dt/2 )
     k4 = F( U + dt * k3,   t + dt   )
 
     return  U + dt * ( k1 + 2*k2 + 2*k3 + k4 )/6


# #############################################################################
# 4. Write a function called Inverse_Euler to integrate one step.

#pensando como en Crank Nicolson busco el implicito de  Euler Un+1=Un + dt*F(Un+1,t)
#llamo a otra funcion que me busque los ceros de la funcion como CN 
# X-U-dt*F(X,t)=0
def Inverse_Euler(U, dt, t, F): 

    def cerosINV(X): 
          return X - U - dt * F(X, t)

    return newton(func = cerosINV, x0 = U ) 

##############################################################################

# 5. Write a function to integrate a Cauchy problem. Temporal scheme, initial
# condition and the function F(U, t) of the Cauchy problem should be input
# arguments.


def integrate_cauchy(EsquemaTemporal, U0, t0, T, dt, F):
    num_steps = int((T - t0) / dt) + 1      #numero pasos
    times = zeros(num_steps)                # tiempo; vector igual al numero de pasos
    states = zeros((len(U0), num_steps))    #matriz u por columnas, con n=numero de pasos de columnas

#condiciones iniciales
    t = t0
    U = U0
    step = 0

    while t <= T:
        times[step] = t # el tiempo va creciendo a T de dt en dt por cada bucle al igual que el paso
        states[:, step] = U          #almacen por columnas de las U soluciones

        U = EsquemaTemporal(U, dt, t, F)
        t =t+ dt
        step =step+ 1

    return times, states

# Elije un esquema temporal (por ejemplo, Euler) y realiza la integracion
times, states = integrate_cauchy(Euler, U0, t0, T, dt, kepler_force)
times_crank, states_crank = integrate_cauchy(Crank_Nicolson, U0, t0, T, dt, kepler_force)
times_rk4, states_rk4 = integrate_cauchy(RK4, U0, t0, T, dt, kepler_force)
times_inverse, states_inverse = integrate_cauchy(Inverse_Euler, U0, t0, T, dt, kepler_force)

# Graficos
plt.figure(1, figsize=(12, 4))

plt.subplot(141)
plt.plot(states[0, :], states[1, :], label='Euler')
plt.title('Orbita - Euler')
plt.axis('equal')

plt.subplot(142)
plt.plot(states_crank[0, :], states_crank[1, :], label='Crank-Nicolson')
plt.title('Orbita - Crank-Nicolson')
plt.axis('equal')

plt.subplot(143)
plt.plot(states_rk4[0, :], states_rk4[1, :], label='RK4')
plt.title('Orbita - RK4')
plt.axis('equal')

plt.subplot(144)
plt.plot(states_inverse[0, :], states_inverse[1, :], label='Inverse Euler')
plt.title('Orbita - Inverse Euler')
plt.axis('equal')

plt.tight_layout()
plt.show()
