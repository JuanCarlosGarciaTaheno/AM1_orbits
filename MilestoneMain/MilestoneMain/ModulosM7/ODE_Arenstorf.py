from numpy import array
import numpy as np

# ODE definition
def Arenstorf(U,t):
    x = U[0]; y = U[1]; Vx = U[2]; Vy = U[3]
    mu_moon = 0.012277471 #Gravitational parameter of the moon
    mu_earth = 1-mu_moon #Gravitational parameter of the Earth
    D1 = np.power(((x+mu_moon)**2 + y**2),1.5)
    D2 = np.power(((x-mu_earth)**2 + y**2),1.5)
    return array([Vx, Vy, x+(2*Vy)-(mu_earth*((x+mu_moon)/D1))-(mu_moon*((x-mu_earth)/D2)), y-(2*Vx)-(mu_earth*(y/D1))-(mu_moon*(y/D2))])             
