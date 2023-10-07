from numpy import array, zeros, linspace
import matplotlib.pyplot as plt
from temporal_schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler
from cauchy_problem import integrate_cauchy
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
# Funcion para la fuerza de Kepler
def kepler_force(U, t):
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    r = (x ** 2 + y ** 2) ** 0.5
    return array([vx, vy, -x / (r ** 3), -y / (r ** 3)])

###################################
# Soluciones
t_euler, states_euler = integrate_cauchy(Euler, U0, custom_t, kepler_force)
t_crank_nicolson, states_crank_nicolson = integrate_cauchy(Crank_Nicolson, U0, custom_t, kepler_force)
t_rk4, states_rk4 = integrate_cauchy(RK4, U0, custom_t, kepler_force)
t_inverse_euler, states_inverse_euler = integrate_cauchy(Inverse_Euler, U0, custom_t, kepler_force)

###################################
# Graficos
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