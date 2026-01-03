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


def binary_string_vector_to_swedish_pump_vector(binary_string):
    return 2 * binary_string - 1


def initialize_population(size_of_population, vrc_length):
    return np.random.randint(2, size=(size_of_population, vrc_length))


def eval_population(population):
    # Optimized: We process individuals one by one but use fast conversion
    score = np.zeros(len(population))

    # 1. Convert entire population matrix from {0,1} to {-1,1} at once
    pop_pump_format = 2 * population - 1

    for i in range(len(population)):
        score[i] = swedish_pump(pop_pump_format[i])

    return score


def pair_according_to_fitness(population, scores):
    population_size = len(population)
    gene_length = population.shape[1]
    fitness_pairs = np.zeros((population_size // 2, 2, gene_length), dtype=int)
    score_sum = np.sum(scores)
    probability_scores = scores / score_sum
    all_indices = np.arange(population_size)

    for i in range(population_size // 2):
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
            # we want to choose two random indexes between 1 and the last index in order to force crossover
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
    # 1. Generate random matrix
    random_matrix = np.random.random(population.shape)
    # 2. Find flips
    flip_mask = random_matrix < p_m
    # 3. Apply flips using XOR (^)
    population[flip_mask] ^= 1
    return population


def traditional_genetic_algorithm(population_size, vector_length, max_calls_to_target_functions):
    # init
    t = 0
    calls = population_size
    population = initialize_population(population_size, vector_length)
    evaluation = eval_population(population)
    best_score = max(evaluation)
    print(f"Best score of the first generation: {best_score}")
    same_score_counter = 0

    # start the run
    while calls < max_calls_to_target_functions:
        # advance current generation
        choose_parents = pair_according_to_fitness(population, evaluation)
        next_gen_population = crossover_population(choose_parents, population_size, p_c=0.8)
        mutate = mutation(next_gen_population, (1 / vector_length))
        evaluation = eval_population(mutate)
        population = mutate
        # see if max generation were found
        same_score_counter += 1
        if max(evaluation) > best_score:
            best_score = max(evaluation)
            print(f"In generation {t + 1}, found a better score: {best_score}")
            same_score_counter = 0
        if same_score_counter > 10 ** 5:
            print("Last generation reached:", t)
            break
        # advance to next generation
        calls += population_size
        t += 1

    return best_score


def calculate_best_score_of_vector_n(current_try, vector_length, population_size):
    print(f"Calculating best score for vector of length {vector_length}:")
    current_best_score = traditional_genetic_algorithm(population_size, vector_length, vector_length * (10 ** 6))
    print(f"Best score for vector of length 25 in try: {current_try} is:", current_best_score)
    return current_best_score


if __name__ == "__main__":
    best_score_25 = 0
    best_score_64 = 0
    best_score_100 = 0

    for i in range(1, 11):
        print("\n\nTry number:", i)

        best_score_25 += calculate_best_score_of_vector_n(i, 25, 200)
        best_score_64 += calculate_best_score_of_vector_n(i, 64, 450)
        best_score_100 += calculate_best_score_of_vector_n(i, 100, 800)

    print("\n\n################################")
    print("FINAL RESULTS")
    print(f"Best score average (L=25): {best_score_25 / 10}")
    print(f"Best score average (L=64): {best_score_64 / 10}")
    print(f"Best score average (L=100): {best_score_100 / 10}")
    print("################################")
