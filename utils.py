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



def binary_to_swedish_pump_format_vector(binary_string):
    return 2 * binary_string - 1


def initialize_population(size_of_population, vrc_length):
    return np.random.randint(2, size=(size_of_population, vrc_length))


def eval_population(population):
    # Optimized: We process individuals one by one but use fast conversion
    score = np.zeros(len(population))

    # convert entire population matrix from {0,1} to {-1,1} at once
    pop_pump_format = binary_to_swedish_pump_format_vector(population)

    for i in range(len(population)):
        curr_score = swedish_pump(pop_pump_format[i])
        score[i] = curr_score

    return score

def pair_according_to_fitness(population, scores, tournament=False):
    population_size = len(population)
    gene_length = population.shape[1]

    empty_parents_arr = np.zeros_like(population)
    selected_parents = tournament_selection(population, scores, empty_parents_arr) if tournament \
                        else roulette_selection(population, scores, empty_parents_arr)

    # reshape to list of pairs
    pairs_number = population_size // 2
    fitness_pairs = selected_parents.reshape(pairs_number, 2, gene_length)

    return fitness_pairs

def roulette_selection(population, scores, selected_parents):
    population_size = len(population)

    score_sum = np.sum(scores)
    probability_scores = scores / score_sum
    all_indices = np.arange(population_size)

    for i in range(population_size // 2):
        j = i * 2
        index_1 = np.random.choice(all_indices, p=probability_scores)
        index_2 = np.random.choice(all_indices, p=probability_scores)
        selected_parents[j] = population[index_1]
        selected_parents[j+1] = population[index_2]

    return selected_parents

def tournament_selection(population, scores, selected_parents):
    population_size = len(population)
    up_bounder = (population_size // 80) + 1
    up_bounder = 3 if up_bounder < 3 else up_bounder    # validate the range

    for i in range(population_size):
        q = np.random.randint(2, up_bounder+1)
        # select q candidate's indexes for current tournament
        candidate_indices  = np.random.randint(0, population_size, size=q)
        candidate_scores = scores[candidate_indices]

        # using argmax to get the index of the highest score among q candidates
        winner_index = candidate_indices[np.argmax(candidate_scores)]
        selected_parents[i] = population[winner_index]

    return selected_parents

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
