import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve


N = 10
n = 41  # Количество узлов коллокации
a = 0
b = 2
A = np.linspace(a, b, n)  # Равномерная сетка
print("Равномерная сетка А:")
print(A, '\n')
h = (b - a) / n  # Шаг сетки
print("Шаг сетки: ", h, '\n')
B = []
for i in range(1, n):
    B.append((A[i-1] + A[i]) / 2)  # Центрально-равномерная сетка
B.append(2)
B = np.array(B)
print("Центрально-равномерная сетка В:")
print(B)


def K(i, j):
    if i == j:
        return (2 - B[i]) * B[i] * h - h**2 / 4
    elif i < j:
        return (2 - B[j]) * B[i] * h
    else:
        return (2 - B[i]) * B[j] * h


# Коэффициенты уравнения
lambda_val = ((-1)**N * (N + 1)) / (4 * N)
beta = ((-1)**N * ((N + 4) / N))
c = (N + 4) / (N + 1)

# Правая часть уравнения
y_values = beta * np.sin(np.pi * B * ((N + 1) / N)) + c * B + 1

# Матрица системы
F = np.zeros((n, n))
K_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        K_matrix[i, j] = K(i, j)
        F[i, j] = (1 if i == j else 0) - lambda_val * K_matrix[i, j]

# Решение системы
x_numerical = solve(F, y_values)

print("Матрица К:")
print(K_matrix)

print("Матрица F:")
print(F)


coeff = (7/5 * np.sin(np.pi / 5) + 39/11 + ((77 * np.pi**2) / (-55 * np.pi**2 + 25)) * np.sin(11*np.pi/5) - np.cos(2 * np.sqrt(11/20))) / np.sin(2 * np.sqrt(11/20))


# Аналитическое решение
def x_analytical_expr(s):
    return np.cos(np.sqrt(11/20) * s) + coeff * np.sin(np.sqrt(11/20) * s) - ((77 * np.pi**2) / (-55 * np.pi**2 + 25)) * np.sin(((11*np.pi)/10) * s)


x_exact = np.array([x_analytical_expr(s) for s in B])

# Метод Фурье
def x_fourier_func(s):
    return 7/5*np.sin(11/10*np.pi*s) + 14/11 * s + 1 + 11/40 * (2.87608 * np.sin(np.pi/2 * s) + 0.093797 * np.sin(np.pi * s) + 0.123969 * np.sin((3 * np.pi)/2 * s))


x_fourier = np.array([x_fourier_func(s) for s in B])

# Графики
plt.figure(figsize=(10, 6))
plt.plot(B, x_numerical, 'bo-', label='Метод коллокаций')
#plt.plot(B, x_fourier, 'g*-', label='Метод Фурье')
plt.plot(B, x_exact, 'r-', label='Аналитическое решение')
plt.xlabel('s')
plt.ylabel('x(s)')
plt.legend()
plt.title('Сравнение решений')
plt.grid()
plt.show()

# Оценка погрешности
error_numerical = np.abs(x_numerical - x_exact).max()
error_fourier = np.abs(x_fourier - x_exact).max()
print(f'Максимальная погрешность метода коллокаций при n = {n}: {error_numerical:.5f}')
print(f'Максимальная погрешность метода Фурье при n = {n}: {error_fourier:.5f}')
