import numpy as np
import inspect

from sympy import *
from datetime import datetime
import matplotlib.pyplot as plt

N = 1000
eps = 1e-5


def plot_graph(coords, f):
    x_min = -10
    y_min = -10
    x_max = 5
    y_max = 5
    delta = 20

    x = np.linspace(x_min, x_max, delta)
    y = np.linspace(y_min, y_max, delta)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    contours = plt.contour(X, Y, Z, 3, colors='black')
    plt.clabel(contours, inline=True, fontsize=12)
    plt.imshow(Z, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='RdGy', alpha=0.5)
    plt.colorbar()

    cx, cy = zip(*coords)
    plt.plot(cx, cy, color='black', marker='o')
    plt.show()


def calc_df(func, params):
    arg_symbols = symbols(inspect.getfullargspec(func).args)
    sym_func = func(*arg_symbols)
    dfs = [lambdify(arg_symbols, sym_func.diff(a)) for a in arg_symbols]
    res = []
    for df in dfs:
        res.append(df(*params))
    return res


def next_interval(fdict, f, a: float, b: float, x1: float, x2: float):
    if x1 not in fdict:
        v1 = f(x1)
        fdict[x1] = v1
    else:
        v1 = fdict[x1]
    if x2 not in fdict:
        v2 = f(x2)
        fdict[x2] = v2
    else:
        v2 = fdict[x2]

    if v1 < v2:
        return fdict, a, x2
    elif v1 > v2:
        return fdict, x1, b
    else:
        return fdict, x1, x2


def b_search(f, a, b, eps=1e-3):
    fdict = {}
    step = 0
    delta = eps / 4

    while abs(a - b) > eps:
        x1 = ((a + b) / 2) - delta
        x2 = ((a + b) / 2) + delta
        fdict, a, b = next_interval(fdict, f, a, b, x1, x2)
        step += 1

    return (a + b) / 2, step, len(fdict)


f1 = lambda x, y: 100 * (y - x) ** 2 + (1 - x) ** 2
f2 = lambda x, y: 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


def conjugate_vecs_method(f, x):
    step = 0
    xs = [x]
    start_time = datetime.now()

    x_prev = x
    w_prev = np.array(-1) * calc_df(f, x)
    p_prev = w_prev
    step += 1

    while abs(np.linalg.norm(w_prev)) > eps and step < N:
        wk = np.array(-1) * calc_df(f, x_prev)
        gamma = (np.linalg.norm(wk) ** 2) / (np.linalg.norm(w_prev) ** 2)
        pk = wk + gamma * p_prev

        psi = lambda chi: f(*(x_prev + chi * p_prev))
        hk, _, _ = b_search(psi, 0, 1, eps)

        xk = x_prev + hk * pk

        x_prev = xk
        w_prev = wk
        p_prev = pk

        step += 1
        xs.append(list(x_prev))

    new_step = 0
    new_xs = []
    if step == N:
        x, new_step, time, new_xs = conjugate_vecs_method(f, xk)

    end_time = datetime.now()
    time = (start_time - end_time).seconds

    return x_prev, step + new_step, time, xs + new_xs


if __name__ == '__main__':
    res, steps, time, coords = conjugate_vecs_method(f1, [0., 0.])
    print('Result:', res)
    print('Iterations number: ', steps)
    print('Duration: ', time)
    plot_graph(coords[0::100], f1)
