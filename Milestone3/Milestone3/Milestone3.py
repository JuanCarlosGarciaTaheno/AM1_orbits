import numpy as np
from numpy import array, linspace
import matplotlib.pyplot as plt
from cauchy_problem import integrate_cauchy
from temporal_schemes import Euler, Inverse_Euler, Crank_Nicolson, RK4

# Funcion que define el problema N-body
def F_Kepler(U, t):
    r, drdt = np.split(U, 2)
    dvdt = -r / np.linalg.norm(r)**3
    return np.concatenate([drdt, dvdt])

# Funcion para evaluar errores de integracion numerica usando extrapolacion de Richardson
# Defino la funcion error_evaluation para evaluar errores de integracion numerica mediante extrapolacion de Richardson.
# Utilizo un esquema temporal especifico (Scheme) y orden de extrapolacion (order).
# Devuelve matrices de error y solucion.
def error_evaluation(Time_Domain, Differential_operator, Scheme, order, U0):
    N = len(Time_Domain) - 1
    t1 = Time_Domain
    t2 = np.zeros(2 * N + 1)
    Error = np.zeros((N + 1, len(U0)))

    # Extrapolacion de Richardson
    for i in range(N):
        t2[2 * i] = t1[i] # el doble t
        t2[2 * i + 1] = (t1[i] + t1[i + 1]) / 2
    t2[2 * N] = t1[N]

    U1 = integrate_cauchy(Differential_operator, t1, U0, Scheme)
    U2 = integrate_cauchy(Differential_operator, t2, U0, Scheme)

    for i in range(N + 1):
        Error[i, :] = (U2[2 * i, :] - U1[i, :]) / (1 - 1. / 2**order)

    Solution = U1 + Error

    return Error, Solution

# Funcion para evaluar la tasa de convergencia de diferentes esquemas temporales
# Defino la funcion temporal_convergence_rate para evaluar la tasa de convergencia de diferentes esquemas temporales
# Utiliza un numero de iteraciones m
# Devuelve el orden de convergencia y registros de error y numero de pasos de tiempo
def temporal_convergence_rate(Time_Domain, Differential_operator, U0, Scheme, m):
    log_E = np.zeros(m) # voy bucando los logaritmos para la representacion del pendiente
    log_N = np.zeros(m) 
    N = len(Time_Domain) - 1
    t1 = Time_Domain
    U1 = integrate_cauchy(Differential_operator, t1, U0, Scheme)

    # Analisis de la tasa de convergencia temporal
    for i in range(m):
        N = 2 * N
        t2 = np.zeros(N + 1)
        t2[0:N + 1:2] = t1
        t2[1:N:2] = (t1[1:int(N / 2) + 1] + t1[0:int(N / 2)]) / 2
        U2 = integrate_cauchy(Differential_operator, t2, U0, Scheme)

        error = np.linalg.norm(U2[N, :] - U1[int(N / 2), :])
        log_E[i] = np.log10(error)
        log_N[i] = np.log10(N)
        t1 = t2
        U1 = U2

    for j in range(m):
        if abs(log_E[j]) > 12:
            break
    j = min(j, m - 1)
    x = log_N[0:j + 1]
    y = log_E[0:j + 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    order = abs(m)
    log_E = log_E - np.log10(1 - 1. / 2**order)

    return order, log_E, log_N

# Funcion para probar la evaluacion de errores
# Define una funcion de prueba test_error_convergence para probar la evaluacion de errores con diferentes esquemas temporales
# Pruebo para los esquemas Euler, Inverse Euler, Crank Nicolson y RK4
def test_error_convergence():
    N = 1000 # si rebajo este valor se hace mas notorio la desviacion
    t = linspace(0, 10, N)
    U0 = array([1, 0, 0, 1])
    order = 1

    #busco lso plot de los errores- busco error y states/cauchy
    print("Error orbita Kepler - Euler")
    Error, U = error_evaluation(t, F_Kepler, Euler, order, U0)

    plt.plot(t, Error[:, 0])
    plt.axis('equal')
    plt.grid()
    plt.show()

    print("Error orbita Kepler - Inverse Euler")
    Error, U = error_evaluation(t, F_Kepler, Inverse_Euler, order, U0)

    plt.plot(t, Error[:, 0])
    plt.axis('equal')
    plt.grid()
    plt.show()

    print("Error orbita Kepler - Crank Nicolson")
    Error, U = error_evaluation(t, F_Kepler, Crank_Nicolson, order, U0)

    plt.plot(t, Error[:, 0])
    plt.axis('equal')
    plt.grid()
    plt.show()

    print("Error orbita Kepler - RK4")
    Error, U = error_evaluation(t, F_Kepler, RK4, order, U0)

    plt.plot(t, Error[:, 0])
    plt.axis('equal')
    plt.grid()
    plt.show()

# Funcion para probar la tasa de convergencia temporal
# Define una funcion de prueba test_temporal_convergence_rate para probar la tasa de convergencia temporal con diferentes esquemas temporales
# Pruebo para los esquemas Euler, Inverse Euler, Crank Nicolson y RK4
def test_temporal_convergence_rate():
    N = 2000 # si lo bajo más se distorsiona una grafica
    t = linspace(0, 10, N)
    U0 = array([1, 0, 0, 1])
    m = 5
    # busco el plot de la convergencia -- los log
    print("Tasa de Convergencia - Euler")
    order, log_e, log_n = temporal_convergence_rate(t, F_Kepler, U0, Euler, m)

    print("orden =", order)
    plt.plot(log_n, log_e)
    plt.axis('equal')
    plt.grid()
    plt.show()

    print("Tasa de Convergencia - Inverse Euler")
    order, log_e, log_n = temporal_convergence_rate(t, F_Kepler, U0, Inverse_Euler, m)

    print("orden =", order)
    plt.plot(log_n, log_e)
    plt.axis('equal')
    plt.grid()
    plt.show()

    print("Tasa de Convergencia - Crank Nicolson")
    order, log_e, log_n = temporal_convergence_rate(t, F_Kepler, U0, Crank_Nicolson, m)

    print("orden =", order)
    plt.plot(log_n, log_e)
    plt.axis('equal')
    plt.grid()
    plt.show()

    print("Tasa de Convergencia - RK4")
    order, log_e, log_n = temporal_convergence_rate(t, F_Kepler, U0, RK4, m)

    print("orden =", order)
    plt.plot(log_n, log_e)
    plt.axis('equal')
    plt.grid()
    plt.show()

# Realizo pruebas utilizando las funciones definidas anteriormente
test_error_convergence()
test_temporal_convergence_rate()


























#####Intento que hice para representar el milestone 3

# from numpy import array, linspace, log
# import matplotlib.pyplot as plt
# from scipy.optimize import newton
# import numpy as np
# ########modulo esuqema temporal###############
# # Funcion para la fuerza de Kepler
# def kepler_force(U, t):
#     x, y, vx, vy = U[0], U[1], U[2], U[3]
#     r = (x ** 2 + y ** 2) ** 0.5
#     return array([vx, vy, -x / (r ** 3), -y / (r ** 3)])

# # Funcion de esquema temporal Euler
# def Euler(U, t1, t2, F):
#     dt = t2 - t1
#     return U + dt * F(U, t1)

# # Funcion de esquema temporal Crank-Nicolson
# def Crank_Nicolson(U, t1, t2, F):
#     def Residual_CN(X):
#         return X - a - (t2 - t1) / 2 * F(X, t2)

#     dt = t2 - t1
#     a = U + (t2 - t1) / 2 * F(U, t1)

#     # Ajusta las condiciones iniciales para mejorar la convergencia
#     initial_guess = U + dt * F(U, t1)

#     try:
#         return newton(Residual_CN, initial_guess, maxiter=100)
#     except RuntimeError:
#         print("Newton's method did not converge. Try adjusting initial conditions.")
#         return U  # Devuelve la solucion actual si no converge

# # Funcion de esquema temporal RK4
# def RK4(U, t1, t2, F):
#     dt = t2 - t1
#     k1 = F(U, t1)
#     k2 = F(U + dt * k1 / 2, t1 + (t2 - t1) / 2)
#     k3 = F(U + dt * k2 / 2, t1 + (t2 - t1) / 2)
#     k4 = F(U + dt * k3, t1 + (t2 - t1))

#     return U + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

# # Funcion de esquema temporal Inverse Euler
# def Inverse_Euler(U, t1, t2, F):
#     dt = t2 - t1
#     def cerosINV(X):
#         return X - U - dt * F(X, t2)

#     # Ajusta las condiciones iniciales para mejorar la convergencia
#     initial_guess = U + 0.1 * dt * F(U, t1)

#     try:
#         return newton(func=cerosINV, x0=initial_guess, maxiter=100)
#     except RuntimeError:
#         print("Newton's method did not converge. Try adjusting initial conditions.")
#         return U  # Devuelve la solucion actual si no converge
# ### fin esquema temporal ###

# ### problema de cauchy para este ejercicio ###
# def integrate_cauchy_temporal_scheme(U0, temporal_scheme, custom_t, kepler_force):
#     dt_ref = custom_t[1] - custom_t[0]
#     num_refinements = 4

#     sizes, errors, rates = richardson_extrapolation(temporal_scheme, U0, kepler_force, custom_t[-1], dt_ref, num_refinements)
    
#     return sizes, errors, rates
# ### fin problema cauchy ###

# ### intento de trabajr con la extrapolacion de richardson 
# # no lo consigo 
# #no es lo que busco, con la inversa de euler ademas no hace nada
# # una vez que no veo por donde tirar, miro lo que ha hecho el profe
# # 
# def richardson_extrapolation(temporal_scheme, U0, F, T, dt_ref, num_refinements):
#     sizes = []
#     errors = []

#     for i in range(num_refinements):
#         dt = dt_ref / 2**i
#         sizes.append(dt)
#         U_fine = temporal_scheme(U0, 0, T, F)
#         U_coarse = temporal_scheme(U0, dt, T, F)
        
#         error = np.linalg.norm(U_fine - U_coarse, ord=np.inf)
#         errors.append(error)
    
#     epsilon = 1e-10
#     rates = []
#     for i in range(len(sizes)-1):
#         if errors[i+1] == 0 or sizes[i+1] == 0:
#             rates.append(0)  # Evitar la division por cero
#         else:
#             rates.append(log(errors[i] / (errors[i+1] + epsilon)) / log((sizes[i] + epsilon) / (sizes[i+1] + epsilon)))
    
#     return sizes, errors, rates

# def convergence_rate(sizes, errors):
#     rates = [log(errors[i] / errors[i+1]) / log(sizes[i] / sizes[i+1]) for i in range(len(sizes)-1)]
#     return rates


# # Condiciones iniciales y configuracion
# U0 = array([1.0, 0.0, 0.0, 1.0])
# T = 1.0
# custom_t = linspace(0, T, 1000)

# # Ejemplo esquema temporal de Euler
# sizes_euler, errors_euler, rate_euler = integrate_cauchy_temporal_scheme(U0, Euler, custom_t, kepler_force)

# # Ejemplo esquema temporal de Crank-Nicolson
# sizes_crank_nicolson, errors_crank_nicolson, rate_crank_nicolson = integrate_cauchy_temporal_scheme(U0, Crank_Nicolson, custom_t, kepler_force)

# # Ejemplo esquema temporal de RK4
# sizes_rk4, errors_rk4, rate_rk4 = integrate_cauchy_temporal_scheme(U0, RK4, custom_t, kepler_force)

# # Ejemplo esquema temporal de Inverse Euler
# sizes_inverse_euler, errors_inverse_euler, rate_inverse_euler = integrate_cauchy_temporal_scheme(U0, Inverse_Euler, custom_t, kepler_force)




# # Resultados y visualizacion
# print("Convergence Rate (Euler):", rate_euler)
# print("Convergence Rate (Crank-Nicolson):", rate_crank_nicolson)
# print("Convergence Rate (RK4):", rate_rk4)
# print("Convergence Rate (Inverse Euler):", rate_inverse_euler)

# plt.figure(figsize=(16, 12))

# plt.subplot(221)
# plt.loglog(sizes_euler, errors_euler, marker='o', linestyle='-', label='Euler')
# plt.title('Richardson Extrapolation - Euler')
# plt.xlabel('Step Size')
# plt.ylabel('Error')
# plt.axis('equal')
# plt.legend()

# plt.subplot(222)
# plt.loglog(sizes_crank_nicolson, errors_crank_nicolson, marker='o', linestyle='-', label='Crank-Nicolson')
# plt.title('Richardson Extrapolation - Crank-Nicolson')
# plt.xlabel('Step Size')
# plt.ylabel('Error')
# plt.axis('equal')
# plt.legend()

# plt.subplot(223)
# plt.loglog(sizes_rk4, errors_rk4, marker='o', linestyle='-', label='RK4')
# plt.title('Richardson Extrapolation - RK4')
# plt.xlabel('Step Size')
# plt.ylabel('Error')
# plt.axis('equal')
# plt.legend()

# plt.subplot(224)
# plt.loglog(sizes_inverse_euler, errors_inverse_euler, marker='o', linestyle='-', label='Inverse Euler')
# plt.title('Richardson Extrapolation - Inverse Euler')
# plt.xlabel('Step Size')
# plt.ylabel('Error')
# plt.axis('equal')
# plt.legend()

# plt.tight_layout()
# plt.show()