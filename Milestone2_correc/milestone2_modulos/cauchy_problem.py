from numpy import zeros

# Define una funcion para integrar el sistema de ecuaciones
def integrate_cauchy(EsquemaTemporal, U0, t, F):
    num_steps = len(t)
    states = zeros((len(U0), num_steps))

    U = U0
    for step in range(num_steps - 1):
        t1 = t[step]
        t2 = t[step + 1]

        states[:, step] = U

        U = EsquemaTemporal(U, t1, t2, F)

    # Asegura que la ultima posicion corresponda al ultimo tiempo
    states[:, -1] = U

    return t, states
