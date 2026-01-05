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


def traditional_genetic_algorithm(population_size, vector_length, max_calls_to_target_functions, tournament=False, elitism=False):
    # init
    t = 0
    calls = population_size

    curr_generation = initialize_population(population_size, vector_length)
    eval_curr_gen = eval_population(curr_generation)

    best_score = max(eval_curr_gen)
    print(f"Best score of the first generation: {best_score}")
    same_score_counter = 0
    prev_generation = curr_generation.copy()

    # start the run
    while calls < max_calls_to_target_functions:
        # advance current generation
        choose_parents = pair_according_to_fitness(curr_generation, eval_curr_gen, tournament)
        population_to_mutate = crossover_population(choose_parents, population_size, p_c=0.8)

        new_generation = mutation(population_to_mutate, (1 / vector_length))
        eval_new_gen = eval_population(new_generation)

        #
        if elitism:
            # combine parent and offspring populations and evaluations into one array for each
            union_generations = np.concatenate([curr_generation, new_generation], axis=0)
            union_eval = np.concatenate([eval_curr_gen, eval_new_gen])

            # sort indices by fitness score in descending order, keep only population_size
            best_idx = np.argsort(union_eval)[::-1][:population_size]

            # take the curr generation and curr generation eval according to those indices
            curr_generation = union_generations[best_idx]
            eval_curr_gen = union_eval[best_idx]

        else:
            curr_generation = new_generation
            eval_curr_gen = eval_new_gen

        # stop condition: identical generations
        if prev_generation is not None and np.array_equal(curr_generation, prev_generation):
            print(f"Stopping: generation {t+1} is identical to generation {t}")
            break

        prev_generation = curr_generation.copy()

        # see if max generation were found
        same_score_counter += 1
        if max(eval_curr_gen) > best_score:
            best_score = max(eval_curr_gen)
            print(f"In generation {t + 1}, found a better score: {best_score}")
            same_score_counter = 0
        if same_score_counter > 10 ** 5:
            print("Last generation reached:", t)
            break
        # advance to next generation
        calls += population_size
        t += 1

    return best_score, calls


def calculate_best_score_of_vector_n(current_try, vector_length, population_size, tournament=False, elitism=True):
    method = "TOURNAMENT" if tournament else "ROULETTE"
    elit = "ELITISM" if elitism else "NO-ELITISM"
    budget = vector_length * (10 ** 6)

    print(f"Calculating best score | Try {current_try}/10 | L={vector_length} | pop={population_size} | {method} | {elit}")

    current_best_score, best_score_num_of_calls = traditional_genetic_algorithm(
        population_size=population_size,
        vector_length=vector_length,
        max_calls_to_target_functions=budget,
        tournament=tournament,
        elitism=elitism
    )

    print("*****************")
    print(f"Summary for try {current_try} | L={vector_length} | pop={population_size} | {method} | {elit}")
    print(f"\tBest score: {current_best_score}")
    print(f"\tCalls used: {best_score_num_of_calls} / {budget} ({best_score_num_of_calls / budget:.2%})")
    print("*****************\n")

    return current_best_score



if __name__ == "__main__":
    # --- ROULETTE RUN ---
    roulette_25_total = 0
    roulette_64_total = 0
    roulette_100_total = 0

    print("############################")
    print("USING ROULETTE")
    print("###########################\n")
    for i in range(1, 11):
        print("\n\nTry number:", i)
        roulette_25_total += calculate_best_score_of_vector_n(i, 25, 200)
        roulette_64_total += calculate_best_score_of_vector_n(i, 64, 450)
        roulette_100_total += calculate_best_score_of_vector_n(i, 100, 800)

    # Calculate Roulette Averages
    avg_roulette_25 = roulette_25_total / 10
    avg_roulette_64 = roulette_64_total / 10
    avg_roulette_100 = roulette_100_total / 10

    # --- TOURNAMENT RUN ---
    tournament_25_total = 0
    tournament_64_total = 0
    tournament_100_total = 0

    print("\n\n############################")
    print("USING TOURNAMENT")
    print("###########################\n")

    for i in range(1, 11):
        print("\n\nTry number:", i)
        tournament_25_total += calculate_best_score_of_vector_n(i, 25, 200, True)
        tournament_64_total += calculate_best_score_of_vector_n(i, 64, 450, True)
        tournament_100_total += calculate_best_score_of_vector_n(i, 100, 800, True)

    # Calculate Tournament Averages
    avg_tournament_25 = tournament_25_total / 10
    avg_tournament_64 = tournament_64_total / 10
    avg_tournament_100 = tournament_100_total / 10

    # --- FINAL COMPARISON ---
    print("\n\n############################################################")
    print("                   FINAL RESULTS COMPARISON")
    print("############################################################")
    print(f"{'Vector Length':<15} | {'Roulette Avg':<20} | {'Tournament Avg':<20}")
    print("-" * 60)
    print(f"{'L=25':<15} | {avg_roulette_25:<20.5f} | {avg_tournament_25:<20.5f}")
    print(f"{'L=64':<15} | {avg_roulette_64:<20.5f} | {avg_tournament_64:<20.5f}")
    print(f"{'L=100':<15} | {avg_roulette_100:<20.5f} | {avg_tournament_100:<20.5f}")
    print("############################################################")