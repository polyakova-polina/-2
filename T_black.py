import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def differential_equations(Y, t):
    # Здесь опишите вашу систему дифференциальных уравнений
    # Верните значения производных по t

    # Например, если у вас есть система уравнений:
    # dx/dt = y
    # dy/dt = -x

    x, y = Y

    # Вычислите производные
    dx_dt = y
    dy_dt = x-x**2

    return [dx_dt, dy_dt]


# Задайте диапазон значений t
t = np.linspace(0, 10, 1000)

# Задайте начальные условия для нескольких траекторий
num_trajectories = 10
initial_conditions = np.random.uniform(low=-1.0, high=1.0, size=(num_trajectories, 2))

# Решите систему дифференциальных уравнений для каждого начального условия
trajectories = []
for init_cond in initial_conditions:
    Y = odeint(differential_equations, init_cond, t)
    trajectories.append(Y)

# Вывод траекторий на фазовой плоскости
for trajectory in trajectories:
    plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.xlabel("x")
plt.ylabel("dx/dt")
plt.title("Phase Portrait")
plt.grid(True)
plt.show()