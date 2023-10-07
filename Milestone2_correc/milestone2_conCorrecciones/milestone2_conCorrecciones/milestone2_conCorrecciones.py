from numpy import array, zeros, linspace
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton

# Milestone 2: Prototypes to integrate orbits with functions.
#Nota:
#Como se vio en clase, lo optimo seria trabajar con diferentes modulos;
# es decir, crear un modulo con todas las funciones de integracion numerica; Euler, CN, RK, Eulerinv, temporal_schemes.py
# luego crear otro con los solucionador cauchy_problem.py
# incluso otro con otras funciones por ejemplo non_lineal_sistems.py con funciones como la de newton, la biseccion

#Este codigo sera uno unico


# Condiciones INICIALES

# Defino condiciones iniciales
U0 = array([1.0, 0.0, 0.0, 1.0])  # Condiciones iniciales (x, y, vx, vy)
t0 = 0.0  # tiempo inicial
T = 10  # periodo o tiempo final (2*pi*1=6.28 para una vuelta)

# Crea un vector de tiempo equiespaciado
num_points = 1000  # Numero de puntos deseados
#custom_t = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 7.0] #varia el dt random, no converge el dt es muy amplio
#EL LIMITE EXPLOTA EN EL 0.01 mas grande ya no converge
custom_t = linspace(t0, T, num_points) #10/1000 el dt=0.01
# si num_points 10000 de t0 a T el dt 10/10000=0.001
# si num_points 100 de t0 a T el dt 10/100=0.1, el problema de cauchy no converge

# Funcion para la fuerza de Kepler
def kepler_force(U, t):
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    r = (x ** 2 + y ** 2) ** 0.5
    return array([vx, vy, -x / (r ** 3), -y / (r ** 3)])
##########################################################################
#               Modulo Temporal_schemes.py

# Funcion de esquema temporal Euler
def Euler(U, t1, t2, F):
    dt = t2 - t1
    return U + dt * F(U, t1)

# Funcion de esquema temporal Crank-Nicolson
def Crank_Nicolson(U, t1, t2, F):
    def Residual_CN(X):
        return X - a - (t2 - t1) / 2 * F(X, t2)

    dt = t2 - t1
    a = U + (t2 - t1) / 2 * F(U, t1)
    return newton(Residual_CN, U)

# Funcion de esquema temporal RK4
def RK4(U, t1, t2, F):
    dt = t2 - t1
    k1 = F(U, t1)
    k2 = F(U + dt * k1 / 2, t1 + (t2 - t1) / 2)
    k3 = F(U + dt * k2 / 2, t1 + (t2 - t1) / 2)
    k4 = F(U + dt * k3, t1 + (t2 - t1))

    return U + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Funcion de esquema temporal Inverse Euler
def Inverse_Euler(U, t1, t2, F):
    dt = t2 - t1
    def cerosINV(X):
        return X - U - dt * F(X, t1)

    return newton(func=cerosINV, x0=U)

# me entro la duda el euler inverso era el euler implicito 
# def implicit_euler(U, t1, t2, F):
#     dt = t2 - t1
#     def residual(X):
#         return X - U - dt * F(X, t2)
    
#     return newton(residual, U)
##########################################################################

##########################################################################
#                  Modulo Cauchy_problem.py
# Define una funcion para integrar el sistema de ecuaciones
def integrate_cauchy(EsquemaTemporal, U0, t, F):   #introduce el esquema temporal con el quieres integrar, 
    #luego las condiciones iniciales, el vector t con el que se definira el dt y 
    #por ultimo que F del problema a resolver
    num_steps = len(t)
    states = zeros((len(U0), num_steps))   #states hace referencia a la U los estados

    U = U0  #inicio en U0
    for step in range(num_steps - 1):
        t1 = t[step]   #donde estan t1 y t2
        t2 = t[step + 1]

        states[:, step] = U

        U = EsquemaTemporal(U, t1, t2, F)

    # Asegura que la ultima posicion corresponda al ultimo tiempo
    states[:, -1] = U

    return t, states
#########################################################################


##########################################################################

#           Soluciones
# Llama a la funcion de integracion con tiempos equiespaciados para cada esquema temporal
t_euler, states_euler = integrate_cauchy(Euler, U0, custom_t, kepler_force)
t_crank_nicolson, states_crank_nicolson = integrate_cauchy(Crank_Nicolson, U0, custom_t, kepler_force)
t_rk4, states_rk4 = integrate_cauchy(RK4, U0, custom_t, kepler_force)
t_inverse_euler, states_inverse_euler = integrate_cauchy(Inverse_Euler, U0, custom_t, kepler_force)
##########################################################################


##########################################################################
#           Graficos
plt.figure(1, figsize=(12, 8))

plt.subplot(221)
plt.plot(states_euler[0, :], states_euler[1, :], label='Euler')
plt.title('Orbita - Euler')
plt.axis('equal')

plt.subplot(222)
plt.plot(states_crank_nicolson[0, :], states_crank_nicolson[1, :], label='Crank-Nicolson')
plt.title('Orbita - Crank-Nicolson')
plt.axis('equal')

plt.subplot(223)
plt.plot(states_rk4[0, :], states_rk4[1, :], label='RK4')
plt.title('Orbita - RK4')
plt.axis('equal')

plt.subplot(224)
plt.plot(states_inverse_euler[0, :], states_inverse_euler[1, :], label='Inverse Euler')
plt.title('Orbita - Inverse Euler')
plt.axis('equal')

plt.tight_layout()
plt.show()