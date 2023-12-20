import numpy as np
from numpy import array, zeros, reshape, linspace
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ODES.Orbits import Kepler
from CauchyProblems.Cauchy_Problem import Cauchy_Problem
from Temporal.Temporal_Schemes import Crank_Nicolson, Euler, RK4, Inverse_Euler, leap_frog

# Constantes físicas aproximadas
G = 6.67430e-11  # Constante gravitatoria en m^3 kg^-1 s^-2

# Masas en kg
m_sol = 1.989e30
m_tierra = 5.972e24
m_marte = 6.39e23
m_luna = 7.342e22

def Initial_positions_and_velocities(Nc, Nb):
    U0 = zeros(2 * Nc * Nb)
    U1 = reshape(U0, (Nb, Nc, 2))
    r0 = reshape(U1[:, :, 0], (Nb, Nc))  # posición 
    v0 = reshape(U1[:, :, 1], (Nb, Nc))  # velocidad

    r0[0, :] = [0, 0, 0]  # Condición inicial Sol
    v0[0, :] = [0, 0, 0]

    r0[1, :] = [147e9, 0, 0]  # Condición inicial Tierra
    v0[1, :] = [0, 29790, 0]

    r0[2, :] = [207e9, 0, 0]  # Condición inicial Marte
    v0[2, :] = [0, 24e3, 0]

    r0[3, :] = [147e9 + 384400e3, 0, 0]  # Condición inicial Luna
    v0[3, :] = [0, 29790 + 1.022e3, 0]

    return U0

#____________Definicion Funcion de los N cuerpos------------------
#-----------------------------------------------------------------
#  dvi/dt = - G m sum_j (ri- rj) / | ri -rj |**3, dridt = vi 
#----------------------------------------------------------------- 

def F_NBody(U, t, Nb, Nc):
    Us = reshape(U, (Nb, Nc, 2))
    F = zeros(len(U))
    dUs = reshape(F, (Nb, Nc, 2))

    r = reshape(Us[:, :, 0], (Nb, Nc))
    v = reshape(Us[:, :, 1], (Nb, Nc))

    drdt = reshape(dUs[:, :, 0], (Nb, Nc))  # derivadas
    dvdt = reshape(dUs[:, :, 1], (Nb, Nc))
    dvdt[:, :] = 0

    for i in range(Nb):
        drdt[i, :] = v[i, :]
        for j in range(Nb):
            if j != i:
                d = r[j, :] - r[i, :]
                dvdt[i, :] = dvdt[i, :] + d[:] * G * ((m_sol if j == 0 else m_tierra if j == 1 else m_marte if j == 2 else m_luna) / np.linalg.norm(d)**3)

    return F

def Integrate_NBP():
    def F(U, t):
        return F_NBody(U, t, Nb, Nc)

    N = 100000
    Nb = 4  # 4 cuerpos: Sol, Tierra, Marte y Luna
    Nc = 3  # Perteneciente a R^3

    t0 = 0
    tf = 1 * 365 * 24 * 60 * 60  # Simulación de 1 año en segundos
    Time = linspace(t0, tf, N + 1)

    U0 = Initial_positions_and_velocities(Nc, Nb)
    U = Cauchy_Problem(Time, RK4, F, U0)

    Us = reshape(U, (N + 1, Nb, Nc, 2))
    r = reshape(Us[:, :, :, 0], (N + 1, Nb, Nc))

     # Plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(Nb):
        if i == 0:
            label = 'Sol'
        elif i == 1:
            label = 'Tierra'
        elif i == 2:
            label = 'Marte'
        else:
            label = 'Luna'
        ax.plot(r[:, i, 0], r[:, i, 1], r[:, i, 2], label=label)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

Integrate_NBP()
############################################################################################

                    ##############################
                    ##                          ##
                    ##  ANIMACION  LUNA-TIERRA  ##
                    ##                          ##
                    ##                          ##
                    ##############################
import numpy as np
from numpy import zeros, reshape, linspace
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from CauchyProblems.Cauchy_Problem import Cauchy_Problem
from Temporal.Temporal_Schemes import RK4

# Constantes fisicas aproximadas
G = 6.67430e-11  # Constante gravitatoria en m^3 kg^-1 s^-2

# Masas en kg
m_tierra = 5.972e24
m_luna = 7.342e22

def Initial_positions_and_velocities(Nc, Nb):
    U0 = zeros(2 * Nc * Nb)
    U1 = reshape(U0, (Nb, Nc, 2))
    r0 = reshape(U1[:, :, 0], (Nb, Nc))  # posicion 
    v0 = reshape(U1[:, :, 1], (Nb, Nc))  # velocidad

    r0[0, :] = [0, 0, 0]  # Condicion inicial Tierra (fija en el origen)
    v0[0, :] = [0, 0, 0]

    r0[1, :] = [384400e3, 0, 0]  # Condicion inicial Luna
    v0[1, :] = [0, 1.022e3, 0]

    return U0

def F_NBody(U, t, Nb, Nc):
    Us = reshape(U, (Nb, Nc, 2))
    F = zeros(len(U))
    dUs = reshape(F, (Nb, Nc, 2))

    r = reshape(Us[:, :, 0], (Nb, Nc))
    v = reshape(Us[:, :, 1], (Nb, Nc))

    drdt = reshape(dUs[:, :, 0], (Nb, Nc))  # derivadas
    dvdt = reshape(dUs[:, :, 1], (Nb, Nc))
    dvdt[:, :] = 0

    for i in range(Nb):
        drdt[i, :] = v[i, :]
        for j in range(Nb):
            if j != i:
                d = r[j, :] - r[i, :]
                dvdt[i, :] = dvdt[i, :] + d[:] * G * ((m_tierra if j == 0 else m_luna) / np.linalg.norm(d)**3)

    return F

def Integrate_NBP():
    def F(U, t):
        return F_NBody(U, t, Nb, Nc)

    N = 200
    Nb = 2  # 2 cuerpos: Tierra (fija) y Luna
    Nc = 3  # Perteneciente a R^3

    t0 = 0
    tf = 28 * 24 * 3600  # Simulacion de 28 dias en segundos
    Time = linspace(t0, tf, N + 1)

    U0 = Initial_positions_and_velocities(Nc, Nb)
    U = Cauchy_Problem(Time, RK4, F, U0)

    Us = reshape(U, (N + 1, Nb, Nc, 2))
    r = reshape(Us[:, :, :, 0], (N + 1, Nb, Nc))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ln1, = ax.plot([], [], [], 'bo--', lw=3, markersize=8)  # Azul para la Tierra
    ln2, = ax.plot([], [], [], 'ro--', lw=3, markersize=8)  # Rojo para la Luna
    traza_tierra, = ax.plot([], [], [], 'b--', lw=1)  # Linea de la traza de la Tierra
    traza_luna, = ax.plot([], [], [], 'r--', lw=1)  # Linea de la traza de la Luna

    ax.set_ylim(-5e8, 5e8)
    ax.set_xlim(-5e8, 5e8)
    ax.set_zlim(-5e8, 5e8)

    traza_tierra_x, traza_tierra_y, traza_tierra_z = [], [], []
    traza_luna_x, traza_luna_y, traza_luna_z = [], [], []

    def animate(i):
        ln1.set_data(r[i, 0, 0], r[i, 0, 1])
        ln1.set_3d_properties(r[i, 0, 2])

        ln2.set_data(r[i, 1, 0], r[i, 1, 1])
        ln2.set_3d_properties(r[i, 1, 2])

        traza_tierra_x.append(r[i, 0, 0])
        traza_tierra_y.append(r[i, 0, 1])
        traza_tierra_z.append(r[i, 0, 2])

        traza_luna_x.append(r[i, 1, 0])
        traza_luna_y.append(r[i, 1, 1])
        traza_luna_z.append(r[i, 1, 2])

        traza_tierra.set_data(traza_tierra_x, traza_tierra_y)
        traza_tierra.set_3d_properties(traza_tierra_z)

        traza_luna.set_data(traza_luna_x, traza_luna_y)
        traza_luna.set_3d_properties(traza_luna_z)

    ani = animation.FuncAnimation(fig, animate, frames=N, interval=50)
    ani.save('orbita_tierra_luna_con_traza.gif', writer='pillow', fps=30)

    plt.show()

# Integrar y generar la animacion
Integrate_NBP()
