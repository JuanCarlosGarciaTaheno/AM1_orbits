from numpy import array
from non_linear_systems import custom_newton

# Funcion de esquema temporal Euler
def Euler(U, t1, t2, F):
    dt = t2 - t1
    return U + dt * F(U, t1)

# Funcion de esquema temporal Crank-Nicolson
def Crank_Nicolson(U, t1, t2, F):
    def Residual_CN(X):
        return X - a - (t2 - t1) / 2 * F(X, t2)

    dt = t2 - t1
    a = U + (t2 - t1) / 2 * F(U, t1)
    return custom_newton(Residual_CN, U)

# Funcion de esquema temporal RK4
def RK4(U, t1, t2, F):
    dt = t2 - t1
    k1 = F(U, t1)
    k2 = F(U + dt * k1 / 2, t1 + (t2 - t1) / 2)
    k3 = F(U + dt * k2 / 2, t1 + (t2 - t1) / 2)
    k4 = F(U + dt * k3, t1 + (t2 - t1))

    return U + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Funcion de esquema temporal Inverse Euler
def Inverse_Euler(U, t1, t2, F):
    dt = t2 - t1
    def cerosINV(X):
        return X - U - dt * F(X, t1)

    return custom_newton(func=cerosINV, x0=U)