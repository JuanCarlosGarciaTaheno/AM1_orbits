import numpy as np
import matplotlib.pyplot as plt
from ODES.Orbits import Kepler
from ODES.Integrate import Integrate_ODE
from ODES.CR3BP import CR3BP, Lagrange_points, Stability_LP
from CauchyProblems.Cauchy_Problem import Cauchy_problem_RK4_emb
from Temporal.Temporal_Schemes import Crank_Nicolson, Euler, RK4, Inverse_Euler, leap_frog, adaptive_RK_emb
from numpy import zeros, linspace, random
from numpy import matmul

# Parámetros del sistema CR3BP
mu = 1.2151e-2

# Configuración de la simulación
N = int(10000)
t = linspace(0, 10, N)
Np = 5  # Número de puntos

# Condiciones iniciales para los puntos
U_0 = zeros([Np, 4])
U_0[0, :] = [0.1, 0, 0, 0]
U_0[1, :] = [0.8, 0.6, 0, 0]
U_0[2, :] = [-0.1, 0, 0, 0]
U_0[3, :] = [0.8, -0.6, 0, 0]
U_0[4, :] = [1.01, 0, 0, 0]

# Calcular puntos de Lagrange
LP = Lagrange_points(U_0, Np, mu)
print("Lagrange Points:")
for i, point in enumerate(LP, start=1):
    print(f"LP{i} =", point)

# Seleccionar punto de Lagrange para estudiar
selected_point = input("Introduce el punto de Lagrange a estudiar: ")
try:
    selected_point_index = int(selected_point)
    if 1 <= selected_point_index <= len(LP):
        selected_lagrange_point = LP[selected_point_index - 1]
        print(f"Lagrange Point {selected_point_index}: {selected_lagrange_point} selected")
    else:
        print("Numero invalido.")
except ValueError:
    print("Invalid input. Introduce un numero.")

# Configuración de condiciones iniciales con perturbación
U0 = zeros(4)
U0[0:2] = LP[selected_point_index - 1, :]
perturbation = 1e-4 * random.random()
U0 = U0 + perturbation

# Función de las ecuaciones para el problema CR3BP
def F(U, t):
    return CR3BP(U, mu)

# Resolver el problema de Cauchy
U, dt_array = Cauchy_problem_RK4_emb(t, adaptive_RK_emb, F, U0)

# Configuración para la estabilidad del punto de Lagrange
U0_stab = zeros(4)
U0_stab[0:2] = LP[selected_point_index - 1, :]
eigenvalues = Stability_LP(U0_stab, mu)
print("Forma de U:", U.shape)

# Plotear los resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(U[:, 0], U[:, 1], '-', color="red", label="Orbita")
ax1.plot(-mu, 0, 'o', color="purple", label="Tierra")
ax1.plot(1 - mu, 0, 'o', color="green", label="Luna")
for i in range(Np):
    ax1.plot(LP[i, 0], LP[i, 1], 'o', label=f"Lagrange {i + 1}")

ax2.plot(U[:, 0], U[:, 1], '-', color="blue", label="Orbita")
ax2.plot(LP[selected_point_index - 1, 0], LP[selected_point_index - 1, 1], 'o', color="black", label=f"Lagrange {selected_point_index}")

fig.suptitle(f"Orbita alrededor del Punto de Lagrange {selected_point_index}", fontsize=16)
ax1.set_title("Vista orbital con soluciones del problema de Cauchy", fontsize=12)
ax2.set_title(f"Orbita alrededor del Punto de Lagrange {selected_point_index}", fontsize=12)

for ax in fig.get_axes():
    ax.set(xlabel='x', ylabel='y')
    ax.grid()
    ax.legend()

plt.tight_layout()
plt.show()
