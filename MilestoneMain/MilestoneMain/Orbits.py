

from numpy import array, power

def Kepler(U,t):
    
    x , y, vx, vy = U
    m = (x ** 2 + y ** 2) ** 0.5
    
    if m == 0:
        # Evita la divisi�n por cero
        return array([vx, vy, 0, 0])
    
    return array ([vx, vy, -x/m, -y/m])

def Arenstorf(U,t):
    
    x, y, vx, vy = U
    mu_moon = 0.012277471                    #Gravitational parameter of the moon
    mu_earth = 1-mu_moon                     #Gravitational parameter of the Earth
    D1 = power(((x+mu_moon)**2 + y**2),1.5)
    D2 = power(((x-mu_earth)**2 + y**2),1.5)
    return array([vx, vy, x+(2*vy)-(mu_earth*((x+mu_moon)/D1))-(mu_moon*((x-mu_earth)/D2)), y-(2*vx)-(mu_earth*(y/D1))-(mu_moon*(y/D2))])             
