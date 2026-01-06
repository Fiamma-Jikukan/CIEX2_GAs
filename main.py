from GeneticAlgorithm import *

def calculate_best_score_of_vector_n(current_try, vector_length, population_size, tournament=False, elitism=True):
    method = "TOURNAMENT" if tournament else "ROULETTE"
    elit = "ELITISM" if elitism else "NO-ELITISM"
    budget = vector_length * (10 ** 4)

    print(f"Calculating best score | Try {current_try}/10 | L={vector_length} | pop={population_size} | {method} | {elit}")

    # CHANGED: call the correct function name + unpack the correct return values
    current_best_score, current_best_vector, best_score_num_of_calls = genetic_algorithm(
        population_size=population_size,
        vector_length=vector_length,
        max_calls_to_target_functions=budget,
        tournament=tournament,
        elitism=elitism
    )

    print("*****************")
    print(f"Summary for try {current_try} | L={vector_length} | pop={population_size} | {method} | {elit}")
    print(f"\tBest vector: {current_best_vector}")
    print(f"\tBest score: {current_best_score}")
    print(f"\tCalls used: {best_score_num_of_calls} / {budget} ({best_score_num_of_calls / budget:.2%})")
    print("*****************\n")

    return current_best_score

if __name__ == "__main__":
    np.set_printoptions(linewidth=10**6)

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
