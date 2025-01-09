import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.integrate import odeint

# t0 = 42482
# t_max = 44964
t0 = 0
t_max = 3000
t = np.linspace(t0, t_max, t_max-1)

#initial value 0
def f(y, t,p,q):
    # return (p*(np.exp(math.e*(t*(p+q)))-1))/(p*np.exp(math.e*(t*(p+q)))+q)
    return p+(q-p)*y-q*y**2

y = odeint(f, 0, t, (0.0002, 0.02))

#initial value 15
# def f(t,p,q):
#     return (p*(np.exp(math.e*(t*(p+q)))+14) + 15*q*np.exp(math.e*(t*(p+q))))/((p+15*q)*np.exp(math.e*(t*(p+q)))-14*q)

# y = f(t,0.0002,0.002)

plt.plot(t,y)
plt.show()
