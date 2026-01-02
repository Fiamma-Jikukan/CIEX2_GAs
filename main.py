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

def initialize_population(size_of_population, vrc_length):
    return np.random.randint(2, size=(size_of_population, vrc_length))

def eval_population(population):
    score = np.zeros(len(population))
    for i in range(len(population)):
        pump_vec = binary_string_vector_to_swedish_pump_vector(population[i])
        val = swedish_pump(pump_vec)
        score[i] = val
    return score

def pair_according_to_fitness(population, scores):
    population_size = len(population)
    gene_length = population.shape[1]
    fitness_pairs = np.zeros((population_size // 2, 2, gene_length), dtype=int)
    score_sum = np.sum(scores)
    probability_scores = scores / score_sum
    all_indices = np.arange(population_size)

    for i in range(population_size//2):
        index_1 = np.random.choice(all_indices, p=probability_scores)
        index_2 = np.random.choice(all_indices, p=probability_scores)
        fitness_pairs[i, 0] = population[index_1]
        fitness_pairs[i, 1] = population[index_2]
    return fitness_pairs

def crossover_population(parents, population_size, p_c):
    """receives parents pairs and returns next gen population"""
    gene_length = parents.shape[2]
    next_gen_population = np.zeros((population_size, gene_length), dtype=int)
    number_of_parents_pairs = len(parents)
    for i in range(number_of_parents_pairs):
        parent_1 = parents[i, 0]
        parent_2 = parents[i, 1]

        if np.random.rand() < p_c:
            points = np.random.choice(np.arange(1, gene_length), size=2, replace=False)
            points.sort()
            pt1, pt2 = points[0], points[1]
            # Head(P1) + Middle(P2) + Tail(P1)
            child_1 = np.concatenate((parent_1[:pt1], parent_2[pt1:pt2], parent_1[pt2:]))
            # Head(P2) + Middle(P1) + Tail(P2)
            child_2 = np.concatenate((parent_2[:pt1], parent_1[pt1:pt2], parent_2[pt2:]))
        else:
            child_1 = parent_1.copy()
            child_2 = parent_2.copy()

        next_gen_population[i * 2] = child_1
        next_gen_population[i * 2 + 1] = child_2
    return next_gen_population


def mutation(population, p_m):
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            bit = population[i, j]
            if np.random.random() < p_m:
                population[i, j] = 1 - population[i, j]



def traditional_genetic_algorithm(population_size, vector_length, max_calls_to_target_functions):
    t = 0
    calls = 0
    population = initialize_population(population_size, vector_length)
    evaluation = eval_population(population)
    while calls < max_calls_to_target_functions:
        choose_parents = pair_according_to_fitness(population, evaluation)
        next_gen_population = crossover_population(choose_parents, population_size, p_c=0.5)
        mutate = mutation(next_gen_population, p_m=0.1)








if __name__ == "__main__":
    vec = np.array([1, 1, -1, -1, 1])
    swedish_to_bin = swedish_pump_vector_to_binary_string(vec)
    bin_to_swedish = binary_string_vector_to_swedish_pump_vector(swedish_to_bin)
    print(swedish_to_bin)
    print(bin_to_swedish)
    popo = initialize_population(20,13)
    print(popo)
    print(eval_population(popo))
