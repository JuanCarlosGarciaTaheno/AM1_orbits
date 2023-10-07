# Funcion personalizada de Newton
#Con esta funcion se reolveran CN y Euler inverso

def custom_newton(func, x0, tol=1e-6, max_iter=100, delta=1e-6):
    def func_derivative(x):
        return (func(x + delta) - func(x - delta)) / (2 * delta)

    x = x0
    for _ in range(max_iter):
        delta_x = func(x) / func_derivative(x)
        x = x - delta_x
        if (abs(delta_x) < tol).any():
            return x
    return x
