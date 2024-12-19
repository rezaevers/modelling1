import numpy as np
import matplotlib.pyplot as plt
import math

# t0 = 42482
# t_max = 44964
t0 = 0
t_max = 3000
t = np.linspace(t0, t_max, t_max-1)

def f(t,p,q):
    return (p*(np.exp(math.e*(t*(p+q)))-1))/(p*np.exp(math.e*(t*(p+q)))+q)

y = f(t,0.0001,0.001)

plt.plot(t,y)
plt.show()
