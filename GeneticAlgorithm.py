from utils import *

def genetic_algorithm(population_size, vector_length, max_calls_to_target_functions, tournament=False, elitism=False, soft_elitism=False):
    # initialize
    t = 0
    calls = population_size

    curr_generation = initialize_population(population_size, vector_length)
    eval_curr_gen = eval_population(curr_generation)

    # track the best solution for individual seen
    curr_best_index = np.argmax(eval_curr_gen)
    best_individual_score = eval_curr_gen[curr_best_index]
    best_individual_vector = curr_generation[curr_best_index].copy()

    print(f"Best score of the first generation: {best_individual_score}")

    # for stopping conditions
    same_score_counter = 1
    prev_generation = curr_generation.copy()

    while calls < max_calls_to_target_functions:
        # selection and crossover
        choose_parents = pair_according_to_fitness(curr_generation, eval_curr_gen, tournament)
        population_to_mutate = crossover_population(choose_parents, population_size, p_c=0.8)

        new_generation = mutation(population_to_mutate, (1 / vector_length))
        eval_new_gen = eval_population(new_generation)

        # final new generation
        if elitism:
            # soft elitism: up to 5% from previous generation can compete against offspring
            k = max(1, int(0.05 * population_size)) if soft_elitism else population_size

            curr_gen_k_idx = np.argsort(eval_curr_gen)[::-1][:k]    # if not soft: all parents
            curr_gen_k = curr_generation[curr_gen_k_idx]
            eval_curr_gen_k = eval_curr_gen[curr_gen_k_idx]

            # combine k parent and offspring populations and evaluations into one array for each
            union_generations = np.concatenate([curr_gen_k, new_generation], axis=0)
            union_eval = np.concatenate([eval_curr_gen_k, eval_new_gen])

            # Natural Selection: parents and offspring compete; keep population_size fittest individuals
            best_idx = np.argsort(union_eval)[::-1][:population_size]
            curr_generation = union_generations[best_idx]
            eval_curr_gen = union_eval[best_idx]

        else:
            curr_generation = new_generation
            eval_curr_gen = eval_new_gen

        # update best if needed
        curr_best_index = np.argmax(eval_curr_gen)
        candidate_score = eval_curr_gen[curr_best_index]

        if candidate_score > best_individual_score:
            best_individual_score = candidate_score
            best_individual_vector = curr_generation[curr_best_index].copy()

            same_score_counter = 0  # initialize counter
            print(f"In generation {t + 1}, found a better score: {best_individual_score}")

        # stop conditions: same score limit exceed
        if same_score_counter > 10 ** 5:
            print(f"Same score limit exceed: last generation reached: {t}")
            break

        # stop condition: identical generations
        if np.array_equal(curr_generation, prev_generation):
            print(f"Stopping: generation {t+1} is identical to generation {t}")
            break

        # advance to next generation
        prev_generation = curr_generation.copy()
        same_score_counter += 1
        calls += population_size
        t += 1

    best_vector_in_swedish_pump = binary_to_swedish_pump_format_vector(best_individual_vector)
    return best_individual_score, best_vector_in_swedish_pump, calls