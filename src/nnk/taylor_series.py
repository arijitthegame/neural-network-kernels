import math
import numpy as np

####### Get tanh coefficients and split them into positive series and a negative series


def get_tanh_coeffs(n, cache):
    if n == 0:
        return 1
    if n in cache:
        return cache[n]

    sum_result = 0
    for k in range(n):
        sum_result += get_tanh_coeffs(k, cache) * get_tanh_coeffs(n - k - 1, cache)

    result = 1 / (2 * n + 1) * sum_result
    cache[n] = result
    return result


def get_positive_tanh_coeffs(n):
    if n % 2 == 0:
        return 0
    elif n % 4 == 1:
        return get_tanh_coeffs(int(n - 1 / 2), {})
    else:
        return 0


def get_negative_tanh_coeffs(n):
    if n % 2 == 0:
        return 0
    elif n % 4 == 3:
        return get_tanh_coeffs(int(n - 1 / 2), {})
    else:
        return 0


######## Get gelu coefficients and split into positive and negative series


def pos_gelu_coeffs(n):
    if n == 1:
        return 1 / math.sqrt(math.pi)
    if n == 0:
        return 0
    if (n > 1) and (n % 2 == 1):
        return 0
    if (n > 1) and (n % 4 == 0):
        return (
            1
            / math.sqrt(math.pi)
            * 1
            / (math.sqrt(2) ** (n - 1) * (n - 1) * np.math.factorial(int(n / 2) - 1))
        )
    if n % 4 == 2:
        return 0


def neg_gelu_coeffs(n):
    if n == 0:
        return 0
    if n == 1:
        return 0
    if n % 2 == 1:
        return 0
    if n % 4 == 2:
        return (
            1
            / math.sqrt(math.pi)
            * 1
            / (math.sqrt(2) ** (n - 1) * (n - 1) * np.math.factorial(int(n / 2) - 1))
        )
    if n % 4 == 0:
        return 0
