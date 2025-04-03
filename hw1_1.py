import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Заданные параметры
alpha = 4
beta = 10/4
k = 41  # Число узлов

A = np.linspace(-1, 1, k)  # Равномерная сетка
h = A[1] - A[0]  # Шаг сетки

F = np.zeros((k, k))
v = np.zeros(k)

# Заполнение матрицы F и вектора v
for i in range(1, k - 1):
    F[i, i - 1] = 1 / h ** 2 - A[i] / (2 * h)  # Левая диагональ
    F[i, i] = -2 / h ** 2 - 4  # Центральная диагональ
    F[i, i + 1] = 1 / h ** 2 + A[i] / (2 * h)  # Правая диагональ

    # Правая часть уравнения
    v[i] = -A[i] ** 3 * alpha + 2 * (6 * alpha - beta) * A[i] ** 2 + (2 * alpha - beta) * (3 * A[i] - 2)

# Граничные условия
F[0, 0] = 1
v[0] = alpha
F[-1, -1] = 1
v[-1] = 3 * alpha + 2 * beta

print(F)
print(v)


def metod_progonki(F, v, k):
    L = np.zeros(k - 1)
    M = np.zeros(k)
    X = np.zeros(k)

    # Прямой ход
    L[0] = -F[0, 1] / F[0, 0]
    M[0] = v[0] / F[0, 0]
    for i in range(1, k - 1):
        denom = F[i, i] + F[i, i - 1] * L[i - 1]
        L[i] = -F[i, i + 1] / denom
        M[i] = (v[i] - F[i, i - 1] * M[i - 1]) / denom

    M[k - 1] = (v[k - 1] - F[k - 1, k - 2] * M[k - 2]) / (F[k - 1, k - 1] + F[k - 1, k - 2] * L[k - 2])

    # Обратный ход
    X[k - 1] = M[k - 1]
    for i in range(k - 2, -1, -1):
        X[i] = M[i] + L[i] * X[i + 1]

    return X


phi_numeric = metod_progonki(F, v, k)

# Аналитическое решение
phi_exact = 4 * A ** 4 + 4 * A ** 3 + 5 / 2 * A ** 2 + 5 / 2 * A + 4

# Оценка погрешности
error = np.abs(phi_numeric - phi_exact)
print(max(error))

# Визуализация
plt.figure(figsize=(8, 6))
plt.plot(A, phi_numeric, 'ro-', label='Численное решение')
plt.plot(A, phi_exact, 'b-', label='Аналитическое решение')
plt.xlabel('τ')
plt.ylabel('φ(τ)')
plt.legend()
plt.grid()
plt.title('Сравнение аналитического и численного решений')
plt.show()

# Оценка погрешности по правилу Рунге
k_double = 2 * k - 1  # Удвоенное число узлов
A_double = np.linspace(-1, 1, k_double)
h_double = A_double[1] - A_double[0]

# Приближенное решение на удвоенной сетке
F_double = np.zeros((k_double, k_double))
v_double = np.zeros(k_double)

for i in range(1, k_double-1):
    F_double[i, i-1] = 1/h_double**2 - A_double[i]/(2*h_double)
    F_double[i, i] = -2/h_double**2 - 4
    F_double[i, i+1] = 1/h_double**2 + A_double[i]/(2*h_double)
    v_double[i] = -A_double[i]**3 * alpha + 2 * (6*alpha - beta) * A_double[i]**2 + (2*alpha - beta) * (3*A_double[i] - 2)

F_double[0, 0] = 1
v_double[0] = alpha
F_double[-1, -1] = 1
v_double[-1] = 3 * alpha + 2 * beta

phi_double = metod_progonki(F_double, v_double, k_double)

runge_error = np.abs(phi_numeric - phi_double[::2]) / 3
print("Оценка погрешности по правилу Рунге:", max(runge_error))

results = pd.DataFrame({'A': A, 'phi_numeric': phi_numeric, 'phi_exact': phi_exact})
print(results)
