import numpy as np
import matplotlib.pyplot as plt

h = 1/16
x0 = 42482
x_max = 44964
y0 = 0


def f(y,p,q):
    return p + (q-p)*y - q*y**2

def calc_ode(h, x0, x_max, y0, p, q):
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

    graph = plt.plot(xs, ys, label=f"p={p}, q={q}")
    print(ys[-1])
    return graph
    # coords = zip(xs,ys)
    # for c in coords:
    #     print(c)


if __name__ == "__main__":
    calc_ode(h, x_max, y0, 0.00001, 0.0008)
    # for i in range(1, 10):
    #     h = 1/(2**(i))
    #     calc_ode(f, h, x_max, y0, 0.1, 0.01)
    #     print(h)
    calc_ode(h, x_max, y0, 0.0001, 0.0005)
    # plt.axis([x0, x_max, 0, 1.5])
    plt.legend(fontsize=15)
    plt.show()

