import numpy as np

def lagrange_basis(x, xi, x_all):
    L = 1
    for xj in x_all:
        if xj != xi:
            L *= (x - xj) / (xi - xj)
    return L

def lagrange_interpolation(x_data, y_data, x_new):
    y_new = np.zeros_like(x_new)
    for xi, yi in zip(x_data, y_data):
        y_new += yi * lagrange_basis(x_new, xi, x_data)
    return y_new

def cubic_spline_interpolation(x, y, x_new):
    n = len(x)
    a = y
    h = np.diff(x)
    alpha = np.zeros(n - 1)
    for i in range(1, n - 1):
        alpha[i] = 3 / h[i] * (a[i + 1] - a[i]) - 3 / h[i - 1] * (a[i] - a[i - 1])

    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    b = np.zeros(n - 1)
    c = np.zeros(n)
    d = np.zeros(n - 1)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    def spline(xi):
        for i in range(n - 1):
            if x[i] <= xi <= x[i + 1]:
                dx = xi - x[i]
                return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
        return None

    return np.array([spline(xi) for xi in x_new])


def chebyshev_nodes(data, n):
    a, b = 0, len(data) - 1
    k = np.arange(0, n)
    x_chebyshev = np.cos((2*k + 1) * np.pi / (2*n))
    x_chebyshev_rescaled = 0.5 * (a + b) + 0.5 * (b - a) * x_chebyshev
    indices = np.round(x_chebyshev_rescaled).astype(int)
    indices = np.clip(indices, 0, len(data) - 1)
    indices = np.unique(indices)    
    return indices


def normal_nodes(a,b,n):
    return np.linspace(a, b, n, dtype=int)