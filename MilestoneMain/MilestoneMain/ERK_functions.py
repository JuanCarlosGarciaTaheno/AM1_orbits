from scipy.optimize import newton
from numpy import zeros, matmul, size, linspace
from numpy.linalg import norm

# Esta funcion realiza un paso del metodo RK para el orden especificado
def RK_stages(order, U1, t, dt, F):
    
    ordenes, Ns, a, b, bs, c = butcher()
    
    Nk = len(U1)
    # k son los valores intermedios basados en los coeficientes del array butcher
    k = zeros([Ns, Nk])
    
    k[0,:] = F(U1, t + c[0]*dt)

    if order == 1:
        for i in range(1, Ns):
            
            Up = U1
            
            for j in range(i):
                
                Up = Up + dt * a[i, j] * k[j, :]
                
            k[i, :] = F(Up, t + c[i]*dt)
            
        # Solucion final
        U2 = U1 + dt * matmul(b, k)

    elif order == 2:
        for i in range(1, Ns):
            
            Up = U1
            
            for j in range(i):
                
                Up = Up + dt * a[i, j] * k[j, :]
                
            k[i, :] = F(Up, t + c[i]*dt)
            
        U2 = U1 + dt * matmul(bs, k)

    return U2

# Esta funcion calcula el tamano de paso adaptativo basandose en el error entre dos soluciones (dU),
# una tolerancia especificada (tol), el paso de tiempo actual (dt), y los ordenes de precision.
def StepSize(dU, tol, dt, ordenes): 
    error = norm(dU)

    if error > tol:
        step_size = dt * (tol/error) ** (1/(ordenes + 1))
    else:
        step_size = dt

    return step_size

# Esta funcion define los coeficientes del array butcher para el metodo RK embebido
def butcher():

    ordenes = [2, 1]
    Ns = 2

    a = zeros([Ns, Ns-1])
    b = zeros([Ns])
    bs = zeros([Ns])
    c = zeros([Ns])

    c = [0., 1.]
    a[0, :] = [0.]
    a[1, :] = [1.]
    b[:] = [1./2, 1./2]
    bs[:] = [1., 0.]

    return ordenes, Ns, a, b, bs, c

