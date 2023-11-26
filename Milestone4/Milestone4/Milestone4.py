import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from temporal_schemes import RK4, Euler, Inverse_Euler, Crank_Nicolson, leap_frog

# Ecuacion diferencial del oscilador lineal
def linear_oscillator(U, t):
    x, v = U[0], U[1]
    return np.array([v, -x])

# # Nueva funcion para este ejercicio la pongo en el temporal schemes
# def leap_frog(U, dt, t, F):
#      U_half = U + dt/2 * F(U, t*dt)
#      return U + dt * F(U_half, (t + 0.5)*dt)


# Funcion para realizar la simulacion y comparar con la solucion analitica
# Voy mirando como representa lel problema cada integracion y la comparo con la solucion analitica
def compare_numerical_analytical(method, U0, F, dt, t_max):
    n = int(t_max / dt)
    tiempo = np.linspace(0, t_max, n)
    
    x_numerical = np.zeros(n)
    x_numerical[0] = U0[0]

    for i in range(1, n):
        U0 = method(U0, dt, i, F)
        x_numerical[i] = U0[0]

    x_analytical = np.cos(tiempo) #solucion analitica del problema

    plt.figure()
    plt.title(f'Comparison - {method.__name__}')
    plt.plot(tiempo, x_numerical, label=f'{method.__name__} (numerical)')
    plt.plot(tiempo, x_analytical, label='Analytical', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Displacement (x)')
    plt.legend()
    plt.show()

# Parametros de la simulacion
U0 = np.array([1, 0])  # Condiciones iniciales
t_max = 10.0            # Tiempo maximo de la simulacion
dt = 0.1                # Paso de tiempo

# Comparar metodos numericos con la solucion analitica
methods = [Euler, Inverse_Euler, leap_frog, Crank_Nicolson, RK4]

for method in methods:
    compare_numerical_analytical(method, U0, linear_oscillator, dt, t_max)
    

#_______REGION DE ESTABILIDAD_________#

# Define la funcion de estabilidad para cada metodo
def stability_function_euler(z):
    return 1 + z

def stability_function_inverse_euler(z):
    return 1 / (1 - z)

def stability_function_leap_frog(z):
    return 1 + z - 0.5 * z**2

def stability_function_crank_nicolson(z):
    return (1 + 0.5 * z) / (1 - 0.5 * z)

def stability_function_runge_kutta(z):
    return 1 + z + 0.5 * z**2 + 1/6 * z**3 + 1/24 * z**4

# DefinO colores desde rojo hasta azul oscuro para ver bien las ganacias
colors = plt.cm.RdBu(np.linspace(0, 1, 9))

# Visualiza las regiones de estabilidad en el plano complejo
def plot_stability_region(stability_function, title):
    x = np.linspace(-5, 5, 800)
    y = np.linspace(-5, 5, 800)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    W = stability_function(Z)

    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.contour(X, Y, np.abs(W), levels=[1], colors='black')

    # Agrega isolineas con colores de rojo a azul oscuro para las ganancias
    for i, gain in enumerate(np.arange(0.9, 0.0, -0.1)):
        isoline = plt.contour(X, Y, np.abs(W) - gain, levels=[0], colors=[colors[i]], linestyles='dashed')
        plt.clabel(isoline, inline=True, fontsize=8, fmt=f'Gain={gain:.1f}')

    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.grid(True)
    plt.show()

# Visualizar las regiones de estabilidad con isolineas
plot_stability_region(stability_function_euler, 'Stability Region - Euler Method')
plot_stability_region(stability_function_inverse_euler, 'Stability Region - Inverse Euler Method')
plot_stability_region(stability_function_leap_frog, 'Stability Region - Leap-Frog Method')
plot_stability_region(stability_function_crank_nicolson, 'Stability Region - Crank-Nicolson Method')
plot_stability_region(stability_function_runge_kutta, 'Stability Region - Runge-Kutta 4th Order')


# En funcion de los autovalores del problema se situaran sn zonas de ganancias diferentes haciendo
# que aumenten o disminuyan de amplitud 