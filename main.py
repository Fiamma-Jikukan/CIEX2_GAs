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





if __name__ == "__main__":
    swedishtry = swedish_pump(np.array([1, 1, -1, 1, 1]))
    print(swedishtry)