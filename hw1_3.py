import numpy as np
import matplotlib.pyplot as plt


def runge_kutta_4(f, A, x_0, h, N):
    u_values = np.zeros(len(A))
    u_values[0] = x_0
    
    for i in range(len(A) - 1):
        t, u_t = A[i], u_values[i]
        w1 = f(t, u_t, N)
        w2 = f(t + h/2, u_t + h/2 * w1, N)
        w3 = f(t + h/2, u_t + h/2 * w2, N)
        w4 = f(t + h, u_t + h * w3, N)
        u_values[i + 1] = u_t + h/6 * (w1 + 2*w2 + 2*w3 + w4)
    
    return u_values


def adams_bashforth_4(f, A, u_values, h, N):
    for i in range(3, len(A) - 1):
        f1, f2, f3, f4 = f(A[i], u_values[i], N), f(A[i-1], u_values[i-1], N), \
                         f(A[i-2], u_values[i-2], N), f(A[i-3], u_values[i-3], N)
        u_values[i + 1] = u_values[i] + h/24 * (55*f1 - 59*f2 + 37*f3 - 9*f4)
    return u_values


def predictor_corrector(f, A, u_values, h, N):
    for i in range(3, len(A) - 1):
        f1, f2, f3, f4 = f(A[i], u_values[i], N), f(A[i-1], u_values[i-1], N), \
                         f(A[i-2], u_values[i-2], N), f(A[i-3], u_values[i-3], N)
        u_predict = u_values[i] + h/24 * (55*f1 - 59*f2 + 37*f3 - 9*f4)  # прогноз
        f_predict = f(A[i+1], u_predict, N)
        u_values[i + 1] = u_values[i] + h/24 * (9*f_predict + 19*f1 - 5*f2 + f3)  # коррекция
    return u_values


N = 10
t_0, t_end, x_0, h  = 0.2, 2.2, N, 0.05
f = lambda t, x, N: 5 * (2 * N + 3) / (N + 2) * np.sin(5 * (3 * (N + 7) + 3) / ((N + 7) * t) * x)
A = np.arange(t_0, t_end + h, h)
print(f"A: {A}")

x_rk4 = runge_kutta_4(f, A, x_0, h, N)
x_ab4 = adams_bashforth_4(f, A, x_rk4.copy(), h, N)
x_pc = predictor_corrector(f, A, x_rk4.copy(), h, N)

print(f"u_rk4: {x_rk4}")
print(f"u_ab4: {x_ab4}")
print(f"u_pc: {x_pc}")

plt.figure(figsize=(10, 6))
plt.plot(A, x_rk4, label='Рунге Кутта', marker='o')
plt.plot(A, x_ab4, label='Адамс-Башфорт', linestyle='--')
plt.plot(A, x_pc, label='Прогноз и коррекция', linestyle=':')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.title('Численные решения задачи Коши')
plt.grid()
plt.show()


h_new = h/2
A_new = np.arange(t_0, t_end + h_new, h_new)
print(f"A_2: {A_new}")

x_rk4_new = runge_kutta_4(f, A_new, x_0, h_new, N)

eps = max(abs(x_rk4[i] - x_rk4_new[2 * i]) for i in range(len(A))) / (2**4 - 1)
print(f"Погрешность по правилу Рунге: Δ = {eps:.6f}")
