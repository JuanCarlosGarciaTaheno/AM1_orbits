import numpy as np
from numpy import array, zeros, reshape, linspace
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cauchy_problem import integrate_cauchy
from temporal_schemes import RK4

# Constantes fisicas aproximadas
G = 6.67430e-11  # Constante gravitatoria en m^3 kg^-1 s^-2

# Masas en kg
m_sol = 1.989e30
m_tierra = 5.972e24
m_marte = 6.39e23
m_luna = 7.342e22

#Defino las condiciones iniciales de los planetas persentes
def Initial_positions_and_velocities(Nc, Nb):
    U0 = zeros(2 * Nc * Nb)
    U1 = reshape(U0, (Nb, Nc, 2))
    r0 = reshape(U1[:, :, 0], (Nb, Nc))  # posicion 
    v0 = reshape(U1[:, :, 1], (Nb, Nc))  # velocidad

    r0[0, :] = [0, 0, 0]  # Condicion inicial Sol
    v0[0, :] = [0, 0, 0]

    r0[1, :] = [147e9, 0, 0] # Condicion inicial Tierra
    v0[1, :] = [0, 29790, 0]

    r0[2, :] = [207e9, 0, 0] # Condicion inicial Marte
    v0[2, :] = [0, 24e3, 0]

    r0[3, :] = [147e9 + 384400e3, 0, 0]  # Posicion inicial de la Luna respecto a la Tierra
    v0[3, :] = [0, 29790 + 1.022e3, 0]   # Velocidad inicial de la Luna respecto a la Tierra

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

    drdt = reshape( dUs[:, :, 0], (Nb, Nc) )  # derivadas
    dvdt = reshape(dUs[:, :, 1], (Nb, Nc))
    dvdt[:, :] = 0
    
    for i in range(Nb):
        drdt[i, :] = v[i, :]
        for j in range(Nb):
            if j != i:
                d = r[j, :] - r[i, :]
                dvdt[i, :] = dvdt[i, :] + d[:] * G * ((m_sol if j == 0 else m_tierra if j == 1 else m_marte if j == 2 else m_luna) / norm(d)**3)

    return F

def Integrate_NBP():
    
    def F(U, t): 
       return F_NBody(U, t, Nb, Nc)
    
    N = 100000
    Nb = 4  # 4 cuerpos: Sol, Tierra, Marte y Luna
    Nc = 3  # Perteneciente a R^3
    Nt = (N+1) * 2 * Nc * Nb
    
    t0 = 0
    tf = 1 * 365 * 24 * 60 * 60  # Simulacion de 1 ano en segundos
    Time = linspace(t0, tf, N + 1)

    U0 = Initial_positions_and_velocities(Nc, Nb)

    U = integrate_cauchy(F, Time, U0, RK4)

    Us = reshape(U, (N + 1, Nb, Nc, 2))
    r = reshape(Us[:, :, :, 0], (N + 1, Nb, Nc))

    ### PLOT 3D ###
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(Nb):
        ax.plot(r[:, i, 0], r[:, i, 1], r[:, i, 2], label=f'Cuerpo {i+1}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.show()

Integrate_NBP()