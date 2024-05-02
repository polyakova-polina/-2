import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy

def lorenzian(G, X0, A):
    return G / (2*np.pi) / ((xm - X0)**2 + G**2 / 4) * A

def clear_lorenzian(G, X0, A):
    return G / (2*np.pi) / ((x - X0)**2 + G**2 / 4) * A

def func(vec):
    line = lorenzian(vec[0], vec[1], vec[2]) + lorenzian(vec[3], vec[4], vec[5])
    return (((ym - line)**2).sum()) ** 0.5 / len(y)

def clear_func(vec):
    line = clear_lorenzian(vec[0], vec[1], vec[2]) + clear_lorenzian(vec[3], vec[4], vec[5])
    return line


def gauss(G, X0, A):
    return 1/(2 * np.pi)**0.5 * np.exp(-(xm - X0)**2 / 2 / G**2) * A

def clear_gauss(G, X0, A):
    return 1/(2 * np.pi)**0.5 * np.exp(-(x - X0)**2 / 2 / G**2) * A

def gfunc(vec):
    line = gauss(vec[0], vec[1], vec[2]) + gauss(vec[3], vec[4], vec[5])
    return (((ym - line)**2).sum()) ** 0.5 / len(y)

def clear_gfunc(vec):
    line = clear_gauss(vec[0], vec[1], vec[2]) + clear_gauss(vec[3], vec[4], vec[5])
    return line


def mnk(x,y):
    le = len(x)

    k = ((x * y).sum() - x.sum() * y.sum() / le) / ((x * x).sum() - (x.sum())**2 / le)
    b = y.sum() / le - k * x.sum() / le
    #sk = 1 / le ** 0.5 * (((y * y).sum() - (y.sum())**2 / le)/((x * x).sum() - (x.sum())**2 / le) - b**2)**0.5
    sk = 1 / le ** 0.5 * (abs(((y * y).sum() - (y.sum()) ** 2 / le) / ((x * x).sum() - (x.sum()) ** 2 / le) - b ** 2)) ** 0.5
    sb = sk * ((x * x).sum() / le - (x.sum())**2 / le**2) ** 0.5
    return k, b, sk, sb

def sred(x,y, par):
    y = y[:len(y) - len(y) % par]
    return x[par//2:len(x) - par//2:par], (np.reshape(y, (len(y) // par, par))).sum(axis=1) / par


data = pd.read_csv('Problem_23_file_005.dat', sep='\t', header=None)
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values
xm, ym = sred(x,y,50)
peak_pos = []
for i in range(2,len(xm) - 2):
    if ym[i] > ym[i-1]  and ym[i] > ym[i+1] and ym[i] > ym.max() / 5:
        peak_pos.append(xm[i])
#print(peak_pos)
res1 = scipy.optimize.minimize(func, np.array([1,peak_pos[0], 1, 1, peak_pos[1], 1]))

yt = clear_func(res1.x)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6, 4))
ax1.scatter(x, y, color='b', s=5)
ax1.scatter(xm, ym, color='r', s = 10)
ax1.scatter(x, yt, color='g', s=5)

noise1 = ax2.hist(y-yt,bins = 50, color = 'b', edgecolor = 'black')




gres1 = scipy.optimize.minimize(gfunc, np.array([1,peak_pos[0], 1, 1, peak_pos[1], 1]))
gyt = clear_gfunc(gres1.x)
fig1, ((a1x1, a1x2), (a1x3, a1x4)) = plt.subplots(2, 2, figsize=(6, 4))
a1x1.scatter(x, y, color='r', s=5)
a1x1.scatter(xm, ym, color='b', s = 10)
a1x1.scatter(x, gyt, color='g', s=5)
gnoise1 = a1x2.hist(y-gyt,bins = 50, color = 'r', edgecolor = 'black')





def if_normal(noise, Ax3, Ax4,parm):
    n_, amp_ = noise[0], noise[1][:-1]
    n = []
    amp = []
    for i in range(len(n_)):
        if n_[i] >2:
            n.append(n_[i])
            amp.append(amp_[i])
    amp = np.array(amp)
    n = np.array(n)
    n = n / n.sum()
    #print(len(n))
    #print(len(n))
    kn,bn, skn, sbn = mnk(amp ** 2, np.log(n))
    sig = (((amp**2 - kn * amp **2 - bn) **2).sum()) ** 0.5 / len(amp) / (-(amp ** 2).min() + (amp ** 2).max())
    #print('sig_par', sig)
    if sig < 0.5 * 100:
        Ax3.scatter(amp ** 2, np.log(n), color='b', s=7)
        lx = np.linspace((amp ** 2).min(), (amp ** 2).max(), 1000)
        ly = kn * lx + bn
        Ax3.scatter(lx, ly, color='r', s=1)
        if parm == 1:
            print('Гауссов шум для пика формы Лоренца')
        else:
            print('Гауссов шум для пика формы Гаусса')
        print('mu = 0')
        sg = (-2 * kn) ** -0.5

        ly = -1/2/sg**2 * lx -np.log(sg * (2 * np.pi)**2)
        Ax3.scatter(lx, ly, color='g', s=1)

        print('sigma = ', sg, 'chek', abs((bn +np.log(sg * (2 * np.pi)**2)) / bn * 100), '%')
        #sg = (-kn / 2) ** -0.5
        Ax4.scatter(amp, n, color='b', s=7)
        #print(n.sum(), (1/ (sg * (2 * np.pi)**0.5) * np.exp(-1 * amp ** 2 / ( 2 * sg**2)) * (-amp_.min() + amp_.max()) / len(amp)).sum())
        Ax4.scatter(amp, 1/ (sg * (2 * np.pi)**0.5) * np.exp(-1 * amp ** 2 / ( 2 * sg**2)) * (-amp.min() + amp.max()) / len(amp), color = 'r', s = 3)

if_normal(noise1, ax3, ax4, 1)

if_normal(gnoise1, a1x3, a1x4, 2)

if ((y - gyt)**2).sum() / ((y - yt)**2).sum() < 1:
    print('Итак, пик Гауссов, тк среднеквадратическое отклонение меньше в', ((y - gyt)**2).sum() / ((y - yt)**2).sum(), 'раз')
if ((y - gyt)**2).sum() / ((y - yt)**2).sum() > 1:
    print('Итак, пик Лоренцнв, тк среднеквадратическое отклонение меньше в', 1 / (((y - gyt)**2).sum() / ((y - yt)**2).sum()), 'раз')

plt.show()
