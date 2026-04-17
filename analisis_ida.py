import time
import random
import math
import tracemalloc
import statistics
import csv
import multiprocessing as mp
import matplotlib.pyplot as plt

from ida_puzzle import NPuzzleIDAStar


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

SIZES = [3, 4, 5, 6, 7, 8]
DIFFICULTIES = {
    "Facil (10)": 10,
    "Medio (20)": 20,
    "Dificil (50)": 50
}

INSTANCES_PER_COMBINATION = 10   # puedes subirlo a 20, 30 o más si quieres
TIMEOUT_SECONDS = 20            # cámbialo a 60 o 120 si tu PC aguanta
RANDOM_SEED = 42


# =========================================================
# UTILIDADES DE TABLERO
# =========================================================

def generate_goal_board(n):
    values = list(range(1, n * n)) + [0]
    return [values[i * n:(i + 1) * n] for i in range(n)]


def board_to_tuple(board):
    return tuple(num for row in board for num in row)


def tuple_to_board(state, n):
    return [list(state[i * n:(i + 1) * n]) for i in range(n)]


def get_valid_blank_moves(zero_idx, n):
    r, c = divmod(zero_idx, n)
    moves = []

    if r > 0:
        moves.append(("U", -1, 0))
    if r < n - 1:
        moves.append(("D", 1, 0))
    if c > 0:
        moves.append(("L", 0, -1))
    if c < n - 1:
        moves.append(("R", 0, 1))

    return moves


def apply_move(state, move, n):
    zero_idx = state.index(0)
    r, c = divmod(zero_idx, n)

    _, dr, dc = move
    nr, nc = r + dr, c + dc
    new_idx = nr * n + nc

    new_state = list(state)
    new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
    return tuple(new_state)


def generate_instance_from_goal(n, scramble_moves):
    """
    Genera una instancia alcanzable desde la meta realizando movimientos válidos
    del espacio en blanco. Evita deshacer inmediatamente el último movimiento.
    """
    goal = generate_goal_board(n)
    state = board_to_tuple(goal)

    last_move = None
    opposite = {"U": "D", "D": "U", "L": "R", "R": "L"}

    for _ in range(scramble_moves):
        zero_idx = state.index(0)
        valid_moves = get_valid_blank_moves(zero_idx, n)

        if last_move is not None:
            valid_moves = [m for m in valid_moves if m[0] != opposite[last_move]]

        move = random.choice(valid_moves)
        state = apply_move(state, move, n)
        last_move = move[0]

    return tuple_to_board(state, n), goal


# =========================================================
# EJECUCIÓN DEL SOLVER CON TIMEOUT
# =========================================================

def solve_instance_worker(start_board, goal_board, n, queue):
    """
    Proceso hijo: resuelve una instancia y manda resultados por Queue.
    """
    try:
        solver = NPuzzleIDAStar(start_board, goal_board, n)

        tracemalloc.start()
        t0 = time.perf_counter()
        solution = solver.ida_star()
        t1 = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        queue.put({
            "solved": solution is not None,
            "time": t1 - t0,
            "nodes": solver.nodes_expanded,
            "moves": len(solution) if solution is not None else None,
            "ram_kb": peak_mem / 1024.0
        })

    except Exception as e:
        queue.put({
            "solved": False,
            "time": None,
            "nodes": None,
            "moves": None,
            "ram_kb": None,
            "error": str(e)
        })


def solve_with_timeout(start_board, goal_board, n, timeout_seconds):
    queue = mp.Queue()
    process = mp.Process(
        target=solve_instance_worker,
        args=(start_board, goal_board, n, queue)
    )

    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "solved": False,
            "timeout": True,
            "time": timeout_seconds,
            "nodes": None,
            "moves": None,
            "ram_kb": None
        }

    if not queue.empty():
        result = queue.get()
        result["timeout"] = False
        return result

    return {
        "solved": False,
        "timeout": False,
        "time": None,
        "nodes": None,
        "moves": None,
        "ram_kb": None
    }


# =========================================================
# EXPERIMENTOS
# =========================================================

def run_experiments():
    random.seed(RANDOM_SEED)
    raw_results = []

    total_cases = len(SIZES) * len(DIFFICULTIES) * INSTANCES_PER_COMBINATION
    case_counter = 0

    print("=" * 70)
    print("INICIANDO ANÁLISIS EMPÍRICO DE IDA*")
    print("=" * 70)
    print(f"Tamaños a probar: {SIZES}")
    print(f"Dificultades: {list(DIFFICULTIES.keys())}")
    print(f"Instancias por combinación: {INSTANCES_PER_COMBINATION}")
    print(f"Timeout por tablero: {TIMEOUT_SECONDS} segundos")
    print("=" * 70)

    for n in SIZES:
        for difficulty_name, scramble_moves in DIFFICULTIES.items():
            for instance_id in range(1, INSTANCES_PER_COMBINATION + 1):
                case_counter += 1
                print(
                    f"[{case_counter}/{total_cases}] "
                    f"Probando {n}x{n} - {difficulty_name} - instancia {instance_id}"
                )

                start_board, goal_board = generate_instance_from_goal(n, scramble_moves)

                result = solve_with_timeout(
                    start_board,
                    goal_board,
                    n,
                    TIMEOUT_SECONDS
                )

                raw_results.append({
                    "n": n,
                    "difficulty": difficulty_name,
                    "scramble_moves": scramble_moves,
                    "instance": instance_id,
                    "solved": result["solved"],
                    "timeout": result["timeout"],
                    "time": result["time"],
                    "nodes": result["nodes"],
                    "solution_moves": result["moves"],
                    "ram_kb": result["ram_kb"]
                })

    return raw_results


# =========================================================
# GUARDAR CSV
# =========================================================

def save_results_csv(results, filename="resultados_ida.csv"):
    fieldnames = [
        "n", "difficulty", "scramble_moves", "instance",
        "solved", "timeout", "time", "nodes", "solution_moves", "ram_kb"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResultados guardados en: {filename}")


# =========================================================
# RESUMEN ESTADÍSTICO
# =========================================================

def summarize_results(results):
    summary = {}

    for n in SIZES:
        summary[n] = {}
        for difficulty_name in DIFFICULTIES.keys():
            subset = [
                r for r in results
                if r["n"] == n and r["difficulty"] == difficulty_name
            ]

            solved_subset = [r for r in subset if r["solved"] and not r["timeout"]]

            success_count = sum(1 for r in subset if r["solved"])
            timeout_count = sum(1 for r in subset if r["timeout"])
            fail_count = len(subset) - success_count

            times = [r["time"] for r in solved_subset if r["time"] is not None]
            nodes = [r["nodes"] for r in solved_subset if r["nodes"] is not None]
            moves = [r["solution_moves"] for r in solved_subset if r["solution_moves"] is not None]
            ram = [r["ram_kb"] for r in solved_subset if r["ram_kb"] is not None]

            summary[n][difficulty_name] = {
                "cases": len(subset),
                "success": success_count,
                "fail": fail_count,
                "timeout": timeout_count,
                "success_rate": (success_count / len(subset) * 100.0) if subset else 0.0,
                "time_mean": statistics.mean(times) if times else None,
                "time_std": statistics.stdev(times) if len(times) > 1 else 0.0 if times else None,
                "nodes_mean": statistics.mean(nodes) if nodes else None,
                "nodes_std": statistics.stdev(nodes) if len(nodes) > 1 else 0.0 if nodes else None,
                "moves_mean": statistics.mean(moves) if moves else None,
                "ram_mean": statistics.mean(ram) if ram else None
            }

    return summary


def print_summary(summary):
    print("\n" + "=" * 70)
    print("RESUMEN ESTADÍSTICO")
    print("=" * 70)

    for n in SIZES:
        print(f"\nTABLERO {n}x{n}")
        for difficulty_name in DIFFICULTIES.keys():
            s = summary[n][difficulty_name]
            print(
                f"  {difficulty_name}: "
                f"éxito={s['success']}/{s['cases']} "
                f"({s['success_rate']:.2f}%), "
                f"timeout={s['timeout']}, "
                f"tiempo_prom={s['time_mean']:.6f}s" if s["time_mean"] is not None
                else f"  {difficulty_name}: éxito={s['success']}/{s['cases']} ({s['success_rate']:.2f}%), timeout={s['timeout']}"
            )


# =========================================================
# GRÁFICAS
# =========================================================

def plot_time(summary):
    plt.figure(figsize=(10, 6))

    for difficulty_name in DIFFICULTIES.keys():
        xs = []
        ys = []
        for n in SIZES:
            t = summary[n][difficulty_name]["time_mean"]
            if t is not None:
                xs.append(n)
                ys.append(t)

        plt.plot(xs, ys, marker="o", label=difficulty_name)

    plt.yscale("log")
    plt.title("Tiempo de ejecución vs tamaño del tablero")
    plt.xlabel("N (dimensión NxN)")
    plt.ylabel("Tiempo promedio (segundos, escala log)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("grafica_tiempos.png", dpi=200)
    plt.show()


def plot_success_fail(summary):
    labels = [f"{n}x{n}" for n in SIZES]
    width = 0.25

    for difficulty_name in DIFFICULTIES.keys():
        success = [summary[n][difficulty_name]["success"] for n in SIZES]
        fail = [summary[n][difficulty_name]["fail"] for n in SIZES]

        x = list(range(len(SIZES)))

        plt.figure(figsize=(10, 6))
        plt.bar([i - width/2 for i in x], success, width=width, label="Resueltos")
        plt.bar([i + width/2 for i in x], fail, width=width, label="No resueltos / timeout")

        plt.title(f"Tasa de éxito para {difficulty_name}")
        plt.xlabel("Dimensión del tablero")
        plt.ylabel("Número de casos")
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        filename = f"grafica_exito_{difficulty_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(filename, dpi=200)
        plt.show()


def plot_nodes(summary):
    plt.figure(figsize=(10, 6))

    for difficulty_name in DIFFICULTIES.keys():
        xs = []
        ys = []
        for n in SIZES:
            val = summary[n][difficulty_name]["nodes_mean"]
            if val is not None:
                xs.append(n)
                ys.append(val)

        plt.plot(xs, ys, marker="o", label=difficulty_name)

    plt.yscale("log")
    plt.title("Nodos expandidos vs tamaño del tablero")
    plt.xlabel("N (dimensión NxN)")
    plt.ylabel("Nodos expandidos promedio (escala log)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("grafica_nodos.png", dpi=200)
    plt.show()


def plot_solution_moves(summary):
    plt.figure(figsize=(10, 6))

    for difficulty_name in DIFFICULTIES.keys():
        xs = []
        ys = []
        for n in SIZES:
            val = summary[n][difficulty_name]["moves_mean"]
            if val is not None:
                xs.append(n)
                ys.append(val)

        plt.plot(xs, ys, marker="o", label=difficulty_name)

    plt.title("Longitud de la solución vs tamaño del tablero")
    plt.xlabel("N (dimensión NxN)")
    plt.ylabel("Movimientos promedio en la solución")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("grafica_movimientos.png", dpi=200)
    plt.show()


def plot_ram(summary):
    plt.figure(figsize=(10, 6))

    for difficulty_name in DIFFICULTIES.keys():
        xs = []
        ys = []
        for n in SIZES:
            val = summary[n][difficulty_name]["ram_mean"]
            if val is not None:
                xs.append(n)
                ys.append(val)

        plt.plot(xs, ys, marker="o", label=difficulty_name)

    plt.yscale("log")
    plt.title("Uso de RAM vs tamaño del tablero")
    plt.xlabel("N (dimensión NxN)")
    plt.ylabel("RAM promedio (KB, escala log)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("grafica_ram.png", dpi=200)
    plt.show()


# =========================================================
# PROGRAMA PRINCIPAL
# =========================================================

def main():
    results = run_experiments()
    save_results_csv(results)

    summary = summarize_results(results)
    print_summary(summary)

    plot_time(summary)
    plot_success_fail(summary)
    plot_nodes(summary)
    plot_solution_moves(summary)
    plot_ram(summary)


if __name__ == "__main__":
    mp.freeze_support()
    main()