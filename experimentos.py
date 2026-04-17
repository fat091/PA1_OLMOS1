import random
import time
import matplotlib.pyplot as plt
from ida_puzzle import NPuzzleIDAStar


def generate_goal(n):
    goal = list(range(1, n*n))
    goal.append(0)
    board = []
    for i in range(n):
        board.append(goal[i*n:(i+1)*n])
    return board


def flatten(board):
    return tuple(num for row in board for num in row)


def tuple_to_board(state, n):
    return [list(state[i*n:(i+1)*n]) for i in range(n)]


def random_moves(board, n, moves):
    state = flatten(board)
    zero = state.index(0)

    for _ in range(moves):
        r, c = divmod(zero, n)

        options = []
        if r > 0: options.append((-1,0))
        if r < n-1: options.append((1,0))
        if c > 0: options.append((0,-1))
        if c < n-1: options.append((0,1))

        dr, dc = random.choice(options)

        nr, nc = r+dr, c+dc
        new_index = nr*n + nc

        state = list(state)
        state[zero], state[new_index] = state[new_index], state[zero]
        state = tuple(state)

        zero = new_index

    return tuple_to_board(state, n)


def run_experiment(n, moves, trials=5):

    times = []
    solved = 0

    goal = generate_goal(n)

    for _ in range(trials):

        start = random_moves(goal, n, moves)

        solver = NPuzzleIDAStar(start, goal, n)

        t0 = time.perf_counter()
        sol = solver.ida_star()
        t1 = time.perf_counter()

        times.append(t1 - t0)

        if sol is not None:
            solved += 1

    return times, solved


def main():

    sizes = [3,4]
    difficulties = {
        "facil":10,
        "medio":20,
        "dificil":50
    }

    results = {}

    for size in sizes:

        results[size] = {}

        for diff, moves in difficulties.items():

            print(f"\nProbando {size}x{size} - {diff}")

            times, solved = run_experiment(size, moves)

            results[size][diff] = (times, solved)

            print("Tiempos:", times)
            print("Soluciones:", solved, "/ 5")


    # ----- GRAFICAS -----

    for size in sizes:

        labels = []
        avg_times = []

        for diff in difficulties:

            times = results[size][diff][0]

            labels.append(diff)
            avg_times.append(sum(times)/len(times))

        plt.figure()
        plt.bar(labels, avg_times)
        plt.title(f"Tiempo promedio IDA* tablero {size}x{size}")
        plt.ylabel("Segundos")
        plt.xlabel("Dificultad")

        plt.show()


if __name__ == "__main__":
    main()