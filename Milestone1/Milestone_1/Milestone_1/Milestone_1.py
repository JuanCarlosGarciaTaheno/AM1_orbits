from numpy import array, zeros
import matplotlib.pyplot as plt 
#@import numpy as np

def F_kepler(U):
    x, y, vx, vy= U[0], U[1], U[2], U[3] #asigno componentes a las variables
    mr=(x**2 + y**2)**1.5
    return array([vx,vy,-x/mr, -y/mr])#velocidad x, velocidad y, ecuacion3 y ecuacion 4
N = 1000 #cantidad de pasos de integracion
# U es un vector de 4 componentes luego array
# si hubiese puesto U = [1 ,0 ,0 ,1] 
#print (type(U)) dando la U como una lista, mientras que con el array es un vector

#############################
####### Metodo Euler ########
U_euler = array([1.0, 0.0, 0.0, 1.0])
dt_euler = 0.01
x_euler = array(zeros(N))
y_euler = array(zeros(N))
x_euler[0] = U_euler[0]
y_euler[0] = U_euler[1]

for i in range(1, N):
    F = F_kepler(U_euler)
    U_euler = U_euler + dt_euler * F
    x_euler[i] = U_euler[0]
    y_euler[i] = U_euler[1]
  


################################
#### METODO CRANK- Nicolson ####

U_crank_nicolson  = array([1.0, 0.0, 0.0, 1.0])
dt_crank_nicolson = 0.01

x_crank_nicolson = array(zeros(N))
y_crank_nicolson = array(zeros(N))
x_crank_nicolson[0] = U_crank_nicolson[0]
y_crank_nicolson[0] = U_crank_nicolson[1]

for i in range(1, N):
    U_n = U_crank_nicolson # Valor en el tiempo tn
    F_n = F_kepler(U_n) # Derivadas en el tiempo tn
    
    U_half = U_n + 0.5 * dt_crank_nicolson * F_n # Valor en el tiempo tn+1/2
    F_half = F_kepler(U_half) # Derivadas en el tiempo tn+1/2
    
    U_crank_nicolson = U_n + dt_crank_nicolson * F_half  # Valor en el tiempo tn+1
    x_crank_nicolson[i] = U_crank_nicolson[0]
    y_crank_nicolson[i] = U_crank_nicolson[1]


#para su calculo, hallo las U y F medias situadas en el paso tn+1/2
# luego se actualiza la U con las del smepaso anterior


#######################################
##############  RK 4  #################


U_rk4 = array([1, 0, 0, 1])
dt_rk4 = 0.01
x_rk4 = array(zeros(N))
y_rk4 = array(zeros(N))
x_rk4[0] = U_rk4[0]
y_rk4[0] = U_rk4[1]

for i in range(1, N):
    U_n = U_rk4 # Valor en el tiempo tn
    
    k1 = dt_rk4 * F_kepler(U_n)
    k2 = dt_rk4 * F_kepler(U_n + 0.5 * k1)
    k3 = dt_rk4 * F_kepler(U_n + 0.5 * k2)
    k4 = dt_rk4 * F_kepler(U_n + k3)
    
    U_rk4 = U_n + (k1 + 2*k2 + 2*k3 + k4) / 6.0 # Valor en el tiempo tn+1
    x_rk4[i] = U_rk4[0]
    y_rk4[i] = U_rk4[1]
#para calcular se hallan 4 derivadas intermedias(k1, k2, k3, k4)
#luego se halla U como la ponderacion de las derivadas
plt.figure(figsize=(12, 8))

plt.subplot(131)
plt.plot(x_euler, y_euler)
plt.title('Metodo de Euler')


plt.subplot(132)
plt.plot(x_crank_nicolson, y_crank_nicolson)
plt.title('Metodo Crank-Nicolson')


plt.subplot(133)
plt.plot(x_rk4, y_rk4)
plt.title('Metodo RK4')


plt.tight_layout()
plt.show()