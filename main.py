import numpy as np

def swedish_pump(b):
    # b is a vector of 1s and -1s
    n = len(b)
    correlation = np.correlate(b, b, mode='full')
    center_index = n - 1
    sidelobes = correlation[center_index + 1:]
    E_b = np.sum(sidelobes ** 2)
    if E_b == 0:
        return float('inf')
    f_b = (n ** 2) / (2 * E_b)
    return f_b


def swedish_pump_vector_to_binary_string(b):
    n = len(b)
    binary_string = np.zeros(n)
    for i in range(n):
        if b[i] == 1:
            binary_string[i] = 1
    return binary_string

def binary_string_vector_to_swedish_pump_vector(binary_string):
    n = len(binary_string)
    swedish_pump_vector = np.ones(n)
    for i in range(n):
        if binary_string[i] == 0:
            swedish_pump_vector[i] = -1
    return swedish_pump_vector

if __name__ == "__main__":
    vec = np.array([1, 1, -1, -1, 1])
    swedish_to_bin = swedish_pump_vector_to_binary_string(vec)
    bin_to_swedish = binary_string_vector_to_swedish_pump_vector(swedish_to_bin)
    print(swedish_to_bin)
    print(bin_to_swedish)