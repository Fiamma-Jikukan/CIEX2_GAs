from GeneticAlgorithm import *

BUDGET = 10**6
NUM_TRIES = 10

# experiment settings
L25, POP25 = 25, 200
L64, POP64 = 64, 450
L100, POP100 = 100, 800

# GA settings
USE_ELITISM = True
USE_SOFT_ELITISM = True


def run_one_try(current_try, vector_length, population_size):
    elit = "ELITISM" if USE_ELITISM else "NO-ELITISM"
    budget = vector_length * BUDGET

    if USE_ELITISM and USE_SOFT_ELITISM:
        k = max(1, int(0.05 * population_size))
        elit = f"SOFT {elit}: {k} parents compete"

    print("----------------------------------------------------------------------------------------")
    print(f"Try {current_try}/10 | Vector Length: {vector_length} | Population Size={population_size} | {elit}")
    print("----------------------------------------------------------------------------------------\n")
    roulette_score = None
    roulette_vector = None
    tournament_score = None
    tournament_vector = None

    for i in range(2):
        is_tournament = (i % 2 == 1)
        method = "TOURNAMENT" if is_tournament else "ROULETTE"
        print(method)
        print("-" * len(method))

        best_score, best_vector, calls_used = genetic_algorithm(
            population_size=population_size,
            vector_length=vector_length,
            max_calls_to_target_functions=budget,
            tournament= is_tournament,
            elitism=USE_ELITISM,
            soft_elitism=USE_SOFT_ELITISM
        )

        print("*****************")
        print(f"Summary")
        print(f"\tBest vector: {best_vector}")
        print(f"\tBest score: {best_score}")
        print(f"\tCalls used: {calls_used} / {budget} ({calls_used / budget:.2%})")
        print("*****************\n")

        if is_tournament:
            tournament_score, tournament_vector = best_score, best_vector
        else:
            roulette_score, roulette_vector = best_score, best_vector

    # comparison for this length
    print("*****************")
    print(f"Comparison Methods | Try {current_try}/{NUM_TRIES} | Vector Length={vector_length}")
    print(f"\tRoulette  | Best score: {roulette_score:.5f} | Best vector: {roulette_vector}")
    print(f"\tTournament| Best score: {tournament_score:.5f} | Best vector: {tournament_vector}")
    print("*****************\n")

    return roulette_score, tournament_score

def run_all_tries():
    roulette_total_25 = 0
    roulette_total_64 = 0
    roulette_total_100 = 0

    tournament_total_25 = 0
    tournament_total_64 = 0
    tournament_total_100 = 0

    for i in range(1, NUM_TRIES + 1):
        print("\n\n========================================")
        print(f"Try number: {i}")
        print("========================================\n")
        r25, t25 = run_one_try(i, L25, POP25)
        r64, t64 = run_one_try(i, L64, POP64)
        r100, t100 = run_one_try(i, L100, POP100)

        print("*****************")
        print(f"Try {i} final comparison (all lengths):")
        print(f"\tL=25  | Roulette: {r25:.5f} | Tournament: {t25:.5f}")
        print(f"\tL=64  | Roulette: {r64:.5f} | Tournament: {t64:.5f}")
        print(f"\tL=100 | Roulette: {r100:.5f} | Tournament: {t100:.5f}")
        print("*****************\n")

        roulette_total_25 += r25
        roulette_total_64 += r64
        roulette_total_100 += r100

        tournament_total_25 += t25
        tournament_total_64 += t64
        tournament_total_100 += t100

    roulette_avgs = (
        roulette_total_25 / NUM_TRIES,
        roulette_total_64 / NUM_TRIES,
        roulette_total_100 / NUM_TRIES
    )

    tournament_avgs = (
        tournament_total_25 / NUM_TRIES,
        tournament_total_64 / NUM_TRIES,
        tournament_total_100 / NUM_TRIES
    )

    return roulette_avgs, tournament_avgs

def print_final_results(roulette_avgs, tournament_avgs):

    print("\n\n############################################################")
    print("                   FINAL RESULTS COMPARISON")
    print("############################################################")
    print(f"{'Vector Length':<15} | {'Roulette Avg':<20} | {'Tournament Avg':<20}")
    print("-" * 60)
    print(f"{'L=25':<15} | {roulette_avgs[0]:<20.5f} | {tournament_avgs[0]:<20.5f}")
    print(f"{'L=64':<15} | {roulette_avgs[1]:<20.5f} | {tournament_avgs[1]:<20.5f}")
    print(f"{'L=100':<15} | {roulette_avgs[2]:<20.5f} | {tournament_avgs[2]:<20.5f}")
    print("############################################################")

if __name__ == "__main__":
    np.set_printoptions(linewidth=10**6)

    roulette_avgs, tournament_avgs = run_all_tries()
    print_final_results(roulette_avgs, tournament_avgs)