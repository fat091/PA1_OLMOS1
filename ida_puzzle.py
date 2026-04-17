import sys
import time
from math import inf


class NPuzzleIDAStar:
    def __init__(self, start_board, goal_board, n):
        self.n = n
        self.start = self.board_to_tuple(start_board)
        self.goal = self.board_to_tuple(goal_board)

        self.goal_positions = {}
        for i, value in enumerate(self.goal):
            self.goal_positions[value] = (i // self.n, i % self.n)

        self.nodes_expanded = 0

    @staticmethod
    def board_to_tuple(board):
        return tuple(num for row in board for num in row)

    def tuple_to_board(self, state):
        return [list(state[i * self.n:(i + 1) * self.n]) for i in range(self.n)]

    def is_goal(self, state):
        return state == self.goal

    def find_zero(self, state):
        return state.index(0)

    def get_successors(self, state):
        """
        Devuelve una lista de sucesores en formato:
        [(nuevo_estado, movimiento), ...]
        movimientos:
        U = el espacio en blanco sube
        D = el espacio en blanco baja
        L = el espacio en blanco va a la izquierda
        R = el espacio en blanco va a la derecha
        """
        successors = []
        zero_idx = self.find_zero(state)
        r, c = divmod(zero_idx, self.n)

        moves = [
            ('U', -1, 0),
            ('D', 1, 0),
            ('L', 0, -1),
            ('R', 0, 1)
        ]

        for move_char, dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.n and 0 <= nc < self.n:
                new_idx = nr * self.n + nc
                new_state = list(state)
                new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
                successors.append((tuple(new_state), move_char))

        return successors

    def manhattan(self, state):
        total = 0
        for idx, value in enumerate(state):
            if value == 0:
                continue
            curr_r, curr_c = divmod(idx, self.n)
            goal_r, goal_c = self.goal_positions[value]
            total += abs(curr_r - goal_r) + abs(curr_c - goal_c)
        return total

    def linear_conflict(self, state):
        """
        Linear Conflict:
        Si dos fichas están en su fila/columna meta, pero en orden invertido,
        agrega una penalización de 2 por conflicto.
        """
        conflicts = 0

        # Conflictos en filas
        for row in range(self.n):
            row_values = []
            for col in range(self.n):
                idx = row * self.n + col
                value = state[idx]
                if value != 0:
                    goal_r, goal_c = self.goal_positions[value]
                    if goal_r == row:
                        row_values.append((col, goal_c))

            for i in range(len(row_values)):
                for j in range(i + 1, len(row_values)):
                    _, goal_c_i = row_values[i]
                    _, goal_c_j = row_values[j]
                    if goal_c_i > goal_c_j:
                        conflicts += 1

        # Conflictos en columnas
        for col in range(self.n):
            col_values = []
            for row in range(self.n):
                idx = row * self.n + col
                value = state[idx]
                if value != 0:
                    goal_r, goal_c = self.goal_positions[value]
                    if goal_c == col:
                        col_values.append((row, goal_r))

            for i in range(len(col_values)):
                for j in range(i + 1, len(col_values)):
                    _, goal_r_i = col_values[i]
                    _, goal_r_j = col_values[j]
                    if goal_r_i > goal_r_j:
                        conflicts += 1

        return 2 * conflicts

    def heuristic(self, state):
        """
        h(n) = Manhattan + Linear Conflict
        """
        return self.manhattan(state) + self.linear_conflict(state)

    def is_solvable(self):
        """
        Verifica si el rompecabezas es resoluble comparando paridades
        entre estado inicial y meta.
        """
        def inversion_count(puzzle_state):
            arr = [x for x in puzzle_state if x != 0]
            inv = 0
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    if arr[i] > arr[j]:
                        inv += 1
            return inv

        start_inv = inversion_count(self.start)
        goal_inv = inversion_count(self.goal)

        if self.n % 2 == 1:
            return (start_inv % 2) == (goal_inv % 2)
        else:
            start_zero_row = self.find_zero(self.start) // self.n
            goal_zero_row = self.find_zero(self.goal) // self.n

            start_blank_from_bottom = self.n - start_zero_row
            goal_blank_from_bottom = self.n - goal_zero_row

            return ((start_inv + start_blank_from_bottom) % 2) == (
                (goal_inv + goal_blank_from_bottom) % 2
            )

    def ida_star(self):
        if self.is_goal(self.start):
            return []

        if not self.is_solvable():
            return None

        threshold = self.heuristic(self.start)
        path = []
        visited = {self.start}

        while True:
            temp = self._search(self.start, 0, threshold, path, visited)
            if temp == "FOUND":
                return path[:]
            if temp == inf:
                return None
            threshold = temp

    def _search(self, state, g, threshold, path, visited):
        self.nodes_expanded += 1

        f = g + self.heuristic(state)
        if f > threshold:
            return f

        if self.is_goal(state):
            return "FOUND"

        minimum = inf

        for next_state, move in self.get_successors(state):
            if next_state in visited:
                continue

            visited.add(next_state)
            path.append(move)

            temp = self._search(next_state, g + 1, threshold, path, visited)

            if temp == "FOUND":
                return "FOUND"

            if temp < minimum:
                minimum = temp

            path.pop()
            visited.remove(next_state)

        return minimum


def read_puzzle_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    if n < 3 or n > 8:
        raise ValueError("La dimensión del tablero debe estar entre 3 y 8.")

    expected_lines = 1 + n + n
    if len(lines) != expected_lines:
        raise ValueError(
            f"Formato inválido. Se esperaban {expected_lines} líneas no vacías y se encontraron {len(lines)}."
        )

    start_board = []
    goal_board = []

    for i in range(1, 1 + n):
        row = list(map(int, lines[i].split(",")))
        if len(row) != n:
            raise ValueError("Una fila del tablero inicial no tiene la dimensión correcta.")
        start_board.append(row)

    for i in range(1 + n, 1 + 2 * n):
        row = list(map(int, lines[i].split(",")))
        if len(row) != n:
            raise ValueError("Una fila del tablero meta no tiene la dimensión correcta.")
        goal_board.append(row)

    validate_boards(start_board, goal_board, n)
    return n, start_board, goal_board


def validate_boards(start_board, goal_board, n):
    start_flat = [num for row in start_board for num in row]
    goal_flat = [num for row in goal_board for num in row]

    expected = list(range(n * n))

    if sorted(start_flat) != expected:
        raise ValueError(
            f"El tablero inicial debe contener exactamente los números de 0 a {n * n - 1}."
        )

    if sorted(goal_flat) != expected:
        raise ValueError(
            f"El tablero meta debe contener exactamente los números de 0 a {n * n - 1}."
        )


def print_board(board, title, n):
    print(title)
    for i in range(n):
        row = board[i * n:(i + 1) * n]
        print(list(row))
    print()


def main():
    if len(sys.argv) != 2:
        print("Uso: python ida_puzzle.py archivo_entrada.txt")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        n, start_board, goal_board = read_puzzle_file(filename)

        solver = NPuzzleIDAStar(start_board, goal_board, n)

        print("=" * 50)
        print(f"RESOLUCIÓN DEL ROMPECABEZAS {n}x{n} CON IDA*")
        print("=" * 50)

        print_board(solver.start, "Tablero inicial:", n)
        print_board(solver.goal, "Tablero meta:", n)

        h0 = solver.heuristic(solver.start)
        print(f"Heurística inicial: {h0}")
        print("Iniciando búsqueda...\n")

        start_time = time.perf_counter()
        solution = solver.ida_star()
        end_time = time.perf_counter()

        print("-" * 50)
        if solution is None:
            print("Estado: SIN SOLUCIÓN")
        elif len(solution) == 0:
            print("Estado: EL TABLERO INICIAL YA ES LA META")
        else:
            print("Estado: SOLUCIÓN ENCONTRADA")
            print(f"Movimientos: {','.join(solution)}")

        print(f"Tiempo total: {end_time - start_time:.6f} segundos")
        print(f"Nodos expandidos: {solver.nodes_expanded}")
        if solution is not None:
            print(f"Cantidad de movimientos: {len(solution)}")
        print("=" * 50)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()