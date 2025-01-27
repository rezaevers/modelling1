import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from euler import calc_ode
import math

data = np.genfromtxt("ParisAgreement.csv", delimiter=';')
data = np.delete(data, 0, 0)
data = np.delete(data, len(data) - 1, 0)

t = data[:,0]
t0 = np.min(t)
t = t - np.repeat(t0, len(t))
y = data[:,1]
y = y/198

#initial value 0
def f(y,t,p,q):
    # return (p*(np.exp(math.e*(t*(p+q)))-1))/(p*np.exp(math.e*(t*(p+q)))+q)
    return p+(q-p)*y-q*y**2

def ode(t, p, q):
    return odeint(f, 0, t, (p, q)).flatten()

def calc_error(t1, y1, t2, y2):
    square_sum = 0
    for i, t in np.ndenumerate(t1):
        i = i[0]
        print(int(t))
        square_sum += (y1[i]-y2[int(t)])**2
        print(t, y1[i], y2[int(t)])
    mse = square_sum/len(t1)
    return mse



params, covar = curve_fit(ode,t,y, p0=[0.0001, 0.001])
p_opt, q_opt = params
print("p, q: ", params)
print("covariance: ", covar)
# t_bass = np.linspace(0, np.max(t), int(np.max(t))-1)
t_bass = np.arange(np.max(t)+1)
y_bass = ode(t_bass, p_opt, q_opt)
print("error: ", calc_error(t, y, t_bass, y_bass))

plt.plot(t, y, 'ro')
plt.plot(t_bass, y_bass, 'b-')
plt.show()
