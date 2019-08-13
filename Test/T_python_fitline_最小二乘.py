#python 最小二乘拟合直线
# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 30, num=50)
y = 0.2*x+[np.random.random() for _ in range(50)]
if __name__ == '__main__':
    plt.figure(figsize=(10, 5), facecolor='w')
    plt.plot(x, y, 'ro', lw=2, markersize=6)
    plt.grid(b=True, ls=':')
    plt.xlabel(u'X', fontsize=16)
    plt.ylabel(u'Y', fontsize=16)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
def Least_squares(x,y):
    x_ = x.mean()
    y_ = y.mean()
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)
    for i in np.arange(50):
        k = (x[i]-x_)* (y[i]-y_)
        m += k
        p = np.square( x[i]-x_ )
        n = n + p
    a = m/n
    b = y_ - a* x_
    return a,b
if __name__ == '__main__':
    a,b = Least_squares(x,y)
    print(a,b)
    y1 = a * x + b
    plt.figure(figsize=(10, 5), facecolor='w')
    plt.plot(x, y, 'ro', lw=2, markersize=6)
    plt.plot(x, y1, 'r-', lw=2, markersize=6)
    plt.grid(b=True, ls=':')
    plt.xlabel(u'X', fontsize=16)
    plt.ylabel(u'Y', fontsize=16)
    plt.show()
