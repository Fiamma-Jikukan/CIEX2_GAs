from utils import *

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