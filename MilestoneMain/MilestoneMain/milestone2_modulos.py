from numpy import array, zeros, linspace
import matplotlib.pyplot as plt
from ODES.Orbits import Kepler
from CauchyProblems.Cauchy_Problem import Cauchy_Problem
from Temporal.Temporal_Schemes import Crank_Nicolson, Euler, RK4, Inverse_Euler
from non_linear_systems import custom_newton

#En este codigo se trbaja con 3 modulos, en este:
#Se impoenne las condiciones iniciales y el espaciado del tiempo
#que funcion F a integrar quiza a futuros seria conveniente crear otro modulo para las F
#soluciones 
#graficos

##################################
# CONDICIONES INICIALES
U0 = array([1.0, 0.0, 0.0, 1.0])  # Condiciones iniciales (x, y, vx, vy)
t0 = 0.0  # tiempo inicial
T = 10  # periodo o tiempo final (2*pi*1=6.28 para una vuelta)

# Crea un vector de tiempo equiespaciado
num_points = 1000  # Numero de puntos deseados
custom_t = linspace(t0, T, num_points)  # 10/1000 el dt=0.01

###################################
# Soluciones
states_euler = Cauchy_Problem(custom_t,Euler, Kepler,U0)
states_crank_nicolson = Cauchy_Problem(custom_t,Crank_Nicolson, Kepler,U0)
states_rk4 = Cauchy_Problem(custom_t,RK4, Kepler,U0)
states_inverse_euler = Cauchy_Problem(custom_t,Inverse_Euler, Kepler,U0)

###################################
# Graficos
print(states_euler[:5, :])
print(states_crank_nicolson[:5, :])
print(states_rk4[:5, :])
print(states_inverse_euler[:5, :])

# Soluciones
states_euler = Cauchy_Problem(custom_t, Euler, Kepler, U0)
states_crank_nicolson = Cauchy_Problem(custom_t, Crank_Nicolson, Kepler, U0)
states_rk4 = Cauchy_Problem(custom_t, RK4, Kepler, U0)
states_inverse_euler = Cauchy_Problem(custom_t, Inverse_Euler, Kepler, U0)

# Graficar los dos primeros valores
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.plot(states_euler[:, 0], states_euler[:, 1], label='Euler')
plt.scatter(states_euler[0, 0], states_euler[0, 1], c='red')  # Punto rojo para el primer valor
plt.title('Orbita - Euler')
plt.axis('equal')

plt.subplot(222)
plt.plot(states_crank_nicolson[:, 0], states_crank_nicolson[:, 1], label='Crank-Nicolson')
plt.scatter(states_crank_nicolson[0, 0], states_crank_nicolson[0, 1], c='red')  # Punto rojo para el primer valor
plt.title('Orbita - Crank-Nicolson')
plt.axis('equal')

plt.subplot(223)
plt.plot(states_rk4[:, 0], states_rk4[:, 1], label='RK4')
plt.scatter(states_rk4[0, 0], states_rk4[0, 1], c='red')  # Punto rojo para el primer valor
plt.title('Orbita - RK4')
plt.axis('equal')

plt.subplot(224)
plt.plot(states_inverse_euler[:, 0], states_inverse_euler[:, 1], label='Inverse Euler')
plt.scatter(states_inverse_euler[0, 0], states_inverse_euler[0, 1], c='red')  # Punto rojo para el primer valor
plt.title('Orbita - Inverse Euler')
plt.axis('equal')

plt.tight_layout()
plt.show()
