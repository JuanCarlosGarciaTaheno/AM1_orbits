from numpy import array, zeros
import matplotlib.pyplot as plt
from scipy.optimize import newton

def F_kepler(U, t):
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x**2 + y**2)**1.5
    return array([vx, vy, -x/mr, -y/mr])

N = 1000  # Cantidad de pasos de integracion

# Inicializa las listas para almacenar las coordenadas x e y para cada metodo
x_euler, y_euler = zeros(N+1), zeros(N+1)
x_crank_nicolson, y_crank_nicolson = zeros(N+1), zeros(N+1)
x_rk4, y_rk4 = zeros(N+1), zeros(N+1)

# Condiciones iniciales
U0 = array([1.0, 0.0, 0.0, 1.0])

# Metodo de Euler
U_euler = U0
x_euler[0] = U_euler[0]
y_euler[0] = U_euler[1]
dt_euler = 0.01

for i in range(1, N+1):
    F = F_kepler(U_euler, i*dt_euler)
    U_euler = U_euler + dt_euler * F
    x_euler[i] = U_euler[0]
    y_euler[i] = U_euler[1]

# Metodo de Crank-Nicolson
def Crank_Nicolson(U, dt, t, F):
    def Residual_CN(X):
        return X - U - dt/2 * (F(X, t) + F(U, t + dt))

    return newton(Residual_CN, U)

U_crank_nicolson = U0
x_crank_nicolson[0] = U_crank_nicolson[0]
y_crank_nicolson[0] = U_crank_nicolson[1]
dt_crank_nicolson = 0.01

for i in range(1, N+1):
    U_crank_nicolson = Crank_Nicolson(U_crank_nicolson, dt_crank_nicolson, i*dt_crank_nicolson, F_kepler)
    x_crank_nicolson[i] = U_crank_nicolson[0]
    y_crank_nicolson[i] = U_crank_nicolson[1]

# Metodo RK4
U_rk4 = U0
x_rk4[0] = U_rk4[0]
y_rk4[0] = U_rk4[1]
dt_rk4 = 0.01

for i in range(1, N+1):
    U_n = U_rk4
    k1 = dt_rk4 * F_kepler(U_n, i*dt_rk4)
    k2 = dt_rk4 * F_kepler(U_n + 0.5 * k1, i*dt_rk4 + 0.5*dt_rk4)
    k3 = dt_rk4 * F_kepler(U_n + 0.5 * k2, i*dt_rk4 + 0.5*dt_rk4)
    k4 = dt_rk4 * F_kepler(U_n + k3, (i+1)*dt_rk4)

    U_rk4 = U_n + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    x_rk4[i] = U_rk4[0]
    y_rk4[i] = U_rk4[1]

# Graficos
plt.figure(figsize=(12, 8))

plt.subplot(131)
plt.plot(x_euler, y_euler)
plt.title('Metodo de Euler')
plt.axis('equal')

plt.subplot(132)
plt.plot(x_crank_nicolson, y_crank_nicolson)
plt.title('Metodo Crank-Nicolson')
plt.axis('equal')

plt.subplot(133)
plt.plot(x_rk4, y_rk4)
plt.title('Metodo RK4')
plt.axis('equal')

plt.tight_layout()
plt.show()