import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from euler import calc_ode
import math

data = np.genfromtxt("ParisTreaty.csv", delimiter=';')
data = np.delete(data, 0, 0)
data = np.delete(data, len(data) - 1, 0)

t = data[:,0]
t0 = np.min(t)
t = t - np.repeat(t0, len(t))
y = data[:,1]
y = y/np.max(y)

def f(t,p,q):
    return (p*(np.exp(math.e*(t*(p+q)))-1))/(p*np.exp(math.e*(t*(p+q)))+q)

params, covar = curve_fit(f,t,y, p0=[0.0001, 0.001])
p_opt, q_opt = params
print("p, q: ", params)
print("covariance: ", covar)
t_bass = np.linspace(0, np.max(t), int(np.max(t))-1)
y_bass = f(t_bass, p_opt, q_opt)

# print(data)
# print(x)

plt.plot(t, y, 'ro')
plt.plot(t_bass, y_bass, 'b-')
# calc_ode(1/16,np.min(t), np.max(t), 0, params[0], params[1])
plt.show()
