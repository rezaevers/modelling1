from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
import math

data = np.genfromtxt("ParisAgreement.csv", delimiter=';')
data = np.delete(data, 0, 0)
data = np.delete(data, len(data) - 1, 0)

t = data[:,0]
t0 = np.min(t)
t = t - np.repeat(t0, len(t))
y = data[:,1]
y = y/198

h = 1
x0 = 0
x_max = 3000
y0 = 0


def f(y,p,q):
    return p + (q-p)*y - q*y**2

def calc_ode(h, x0, x_max, y0, p, q, label=""):
    if label == "":
        label = f"p={p}, q={q}"
    xs = np.array([x0])
    ys = np.array([y0])
    x = x0
    y = y0

    while x < x_max:
        xn = x + h
        yn = y + f(y, p, q) * h
        xs = np.append(xs, xn)
        ys = np.append(ys, yn)
        x=xn
        y=yn

    graph = plt.plot(xs, ys, label=label)
    print(ys[-1])
    return graph
    # coords = zip(xs,ys)
    # for c in coords:
    #     print(c)


if __name__ == "__main__":
    plt.plot(t, y, 'ro')
    calc_ode(h, x0, x_max, y0, 0.00063092, 0.00607882)
    # calc_ode(h, x0, x_max, y0, 0.07, 0.007, label="p>q")
    # calc_ode(h, x0, x_max, y0, 0.007, 0.07, label="q>p")
    # for i in range(1, 10):
    #     h = 1/(2**(i))
    #     calc_ode(f, h, x_max, y0, 0.1, 0.01)
    #     print(h)
    # calc_ode(h, x0, x_max, y0, 0.001, 0.005)
    # plt.axis([x0, x_max, 0, 1.5])
    plt.legend(fontsize=15)
    plt.show()

