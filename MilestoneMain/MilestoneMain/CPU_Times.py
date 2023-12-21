from numpy import array, zeros, linspace
import numpy as np
import matplotlib.pyplot as plt
from Orbits import Arenstorf
from Cauchy_Problem import Cauchy_Problem#, Cauchy_problem_LP, Cauchy_Problem_GBS, Cauchy_problem_RK4_emb
from Temporal_Schemes import Euler, GBS_solution_NL, Crank_Nicolson, RK4, ERK,Leap_Frog
from time import process_time
import random

def CPU_Times():
    U0 = array([0.994, 0, 0, -2.00158510637908252240537862224])

    #Parameters definition
    T = 17.0652165601579625588917206249
    N = 100000
    dt = T/N
    t = linspace(0,T,N)
    x = array(zeros(N))
    y = array(zeros(N))
    x[0] = U0[0]
    y[0] = U0[1]

    scheme = array([Crank_Nicolson,Euler,RK4,GBS_solution_NL,Leap_Frog,ERK])
    start = array(zeros(len(scheme)))
    finish = array(zeros(len(scheme)))
    CPU_time = array(zeros(len(scheme)))
    for i in range(len(scheme)):
        start[i] = process_time()
        if  scheme[i] == GBS_solution_NL:
            sol = Cauchy_Problem(t, scheme[i], Arenstorf, U0, 4)
        else:
            sol = Cauchy_Problem(t, scheme[i], Arenstorf, U0)
        finish[i] = process_time()
        CPU_time [i] = finish[i]-start[i]
        print(f"{scheme[i].__name__}, CPU Time =",CPU_time [i]," seconds.")
        
    plt.figure(figsize=(8, 6))  # Tamano del grafico (opcional)
   # colores = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(scheme))]
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    scheme_labels = ['Crank_Nicolson','Euler','RK4','GBS {N = 4}','Leap-Frog','Runge-Kutta Embebido']

    plt.bar(scheme_labels, CPU_time, color=colores)  # Crear barras con los datos

    plt.xlabel('Esquema Temporal')  # Etiqueta para el eje x
    plt.ylabel('CPU_Time [s]')  # Etiqueta para el eje y
    plt.title('Grafico de Barras del Tiempo de Computacion')  # Titulo del grafico

    plt.grid(True)  # Agregar una cuadricula (opcional)
    plt.tight_layout()  # Ajustar diseno

    plt.show()  # Mostrar el grafico


