# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 48:
# 99242 Joana Pereira Ehrhardt Serra da Silva
# 99331 Tiago Alexandre Pereira Antunes

import sys
import numpy as np
import utils
from sys import stdin
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
    compare_searchers,
    compare_graph_searchers,
)

class Board:
    """Representação interna de um tabuleiro de Takuzu."""

    def __init__(self, array):
        self.array = array

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.array[row, col]

    def get_dimension_len(self, dimension: int) -> int:
        return self.array.shape[dimension]

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        return (self.get_number(row+1, col) if row < (self.get_dimension_len(0)-1) else None, self.get_number(row-1, col) if row > 0 else None)

    def top_adjacent_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        return (self.get_number(row+2, col) if row < (self.get_dimension_len(0)-2) else None, self.get_number(row+1, col) if row < (self.get_dimension_len(0)-1) else None)

    def bottom_adjacent_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        return (self.get_number(row-1, col) if row > 0 else None, self.get_number(row-2, col) if row > 1 else None)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        return (self.get_number(row, col-1) if col > 0 else None, self.get_number(row, col+1) if col < (self.get_dimension_len(1)-1) else None)

    def left_adjacent_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        return (self.get_number(row, col-2) if col > 1 else None, self.get_number(row, col-1) if col > 0 else None)

    def right_adjacent_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        return (self.get_number(row, col+1) if col < (self.get_dimension_len(1)-1) else None, self.get_number(row, col+2) if col < (self.get_dimension_len(1)-2) else None)

    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 takuzu.py < input_T01

            > from sys import stdin
            > stdin.readline()
        """
        line = stdin.readline()
        dimension = eval(line[:-1])
        board = np.empty((0, dimension), int)

        for _ in range(dimension):
            line = stdin.readline()
            line_array = [np.fromstring(line, int, sep='\t')]
            board = np.append(board, line_array, axis=0)
        
        return Board(board)

    def resulting_board(self, row: int, col: int, val: int):
        new_array = np.copy(self.array)
        new_array[row, col] = val

        return Board(new_array)

    def __repr__(self) -> str:
        return '\n'.join(['\t'.join(['{}'.format(number) for number in row]) for row in self.array])

    def is_full(self):
        return 2 not in self.array

    def is_valid(self):
        dimension_zero = self.get_dimension_len(0)
        dimension_one = self.get_dimension_len(1)
        for i in range(dimension_zero):
            sumLine = 0
            full = True
            for j in range(dimension_one):
                num = self.get_number(i, j)
                if num == 2:
                    full = False
                if num != 2:
                    sumLine += num
                    if (j > 0 and j < dimension_one-1):
                        adjacent = self.adjacent_horizontal_numbers(i, j)
                        if (num == adjacent[0] and num == adjacent[1]):
                            return False

            if full:
                if (dimension_one % 2 == 0):
                    if (sumLine != dimension_one//2):
                        return False
                else:
                    if (sumLine != (dimension_one//2)+1 and sumLine != (dimension_one//2)):
                        return False

                for n in range(i+1, dimension_zero):
                    if np.array_equal(self.array[i], self.array[n]):
                        return False

            else:
                if (dimension_one % 2 == 0):
                    if (sumLine > dimension_one//2):
                        return False
                else:
                    if (sumLine > (dimension_one//2)+1):
                        return False
            
        return True


    def adjacent_rule(self):
        dimension_zero = board.get_dimension_len(0)
        dimension_one = board.get_dimension_len(1)
        for i in range(dimension_zero):
            for j in range(dimension_one):
                if self.get_number(i, j) == 2:
                    adjacent_horizontal = self.adjacent_horizontal_numbers(i, j)
                    if adjacent_horizontal == (0, 0):
                        return (i, j, 1)
                    elif adjacent_horizontal == (1, 1):
                        return (i, j, 0)
                    left_horizontal = self.left_adjacent_numbers(i, j)
                    if left_horizontal == (0, 0):
                        return (i, j, 1)
                    elif left_horizontal == (1, 1):
                        return (i, j, 0)
                    right_horizontal = self.right_adjacent_numbers(i, j)
                    if right_horizontal == (0, 0):
                        return (i, j, 1)
                    elif right_horizontal == (1, 1):
                        return (i, j, 0)

                    adjacent_vertical = self.adjacent_vertical_numbers(i, j)
                    if adjacent_vertical == (0, 0):
                        return (i, j, 1)
                    elif adjacent_vertical == (1, 1):
                        return (i, j, 0)
                    top_vertical = self.top_adjacent_numbers(i, j)
                    if top_vertical == (0, 0):
                        return (i, j, 1)
                    elif top_vertical == (1, 1):
                        return (i, j, 0)
                    bottom_vertical = self.bottom_adjacent_numbers(i, j)
                    if bottom_vertical == (0, 0):
                        return (i, j, 1)
                    elif bottom_vertical == (1, 1):
                        return (i, j, 0)
        
        return False

    def adjacent_rule_h(self):
        dimension_zero = board.get_dimension_len(0)
        dimension_one = board.get_dimension_len(1)
        total = 0
        l = 0
        for i in range(dimension_zero):
            for j in range(dimension_one):
                if self.get_number(i, j) == 2:
                    total += 1
                    adjacent_horizontal = self.adjacent_horizontal_numbers(i, j)
                    if adjacent_horizontal == (0, 0):
                        l += 1
                        continue
                    elif adjacent_horizontal == (1, 1):
                        l += 1
                        continue
                    left_horizontal = self.left_adjacent_numbers(i, j)
                    if left_horizontal == (0, 0):
                        l += 1
                        continue
                    elif left_horizontal == (1, 1):
                        l += 1
                        continue
                    right_horizontal = self.right_adjacent_numbers(i, j)
                    if right_horizontal == (0, 0):
                        l += 1
                        continue
                    elif right_horizontal == (1, 1):
                        l += 1
                        continue

                    adjacent_vertical = self.adjacent_vertical_numbers(i, j)
                    if adjacent_vertical == (0, 0):
                        l += 1
                        continue
                    elif adjacent_vertical == (1, 1):
                        l += 1
                        continue
                    top_vertical = self.top_adjacent_numbers(i, j)
                    if top_vertical == (0, 0):
                        l += 1
                        continue
                    elif top_vertical == (1, 1):
                        l += 1
                        continue
                    bottom_vertical = self.bottom_adjacent_numbers(i, j)
                    if bottom_vertical == (0, 0):
                        l += 1
                        continue
                    elif bottom_vertical == (1, 1):
                        l += 1
                        continue
        
        return total-l

    def sum_rule_lines_columns(self):
        res = self.sum_rule()
        if res:
            return res
        res = Board(np.transpose(self.array)).sum_rule()
        if res:
            return (res[1], res[0], res[2])
        return False

    def sum_rule(self):
        dimension_zero = self.get_dimension_len(0)
        dimension_one = self.get_dimension_len(1)
        for i in range(dimension_zero):
            num_zeros = 0
            num_ones = 0
            free_positions = []
            for j in range(dimension_one):
                num = self.get_number(i, j)
                if num == 0:
                    num_zeros += 1
                elif num == 1:
                    num_ones += 1
                else:
                    free_positions += [(i, j)]
            if dimension_zero % 2 == 0:
                if num_ones == dimension_zero//2:
                    if len(free_positions) > 0:
                        return (free_positions[0][0], free_positions[0][1], 0)
                elif num_zeros == dimension_zero//2:
                    if len(free_positions) > 0:
                        return (free_positions[0][0], free_positions[0][1], 1)
            else:
                if num_ones == dimension_zero//2 + 1:
                    if len(free_positions) > 0:
                        return (free_positions[0][0], free_positions[0][1], 0)
                elif num_zeros == dimension_zero//2 + 1:
                    if len(free_positions) > 0:
                        return (free_positions[0][0], free_positions[0][1], 1)

        return False

    def get_first_available_actions(self):
        dimension_zero = board.get_dimension_len(0)
        dimension_one = board.get_dimension_len(1)
        for i in range(dimension_zero):
            for j in range(dimension_one):
                if self.get_number(i, j) == 2:
                    return [(i, j, 0), (i, j, 1)]
        return False

    
    

class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    def fill(self, action):
        return TakuzuState(self.board.resulting_board(action[0], action[1], action[2]))

    def is_goal_state(self) -> bool:
        return self.board.is_full() and self.board.is_valid() and Board(np.transpose(self.board.array)).is_valid()

    def adjacent_rule_actions(self):
        return self.board.adjacent_rule()

    def sum_rule_actions(self):
        return self.board.sum_rule_lines_columns()

    def get_first_empty_position_actions(self):
        return self.board.get_first_available_actions()

    def get_h(self):
        return self.board.adjacent_rule_h()


       

class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = TakuzuState(board)

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        if not state.board.is_valid():
            return []
        elif not Board(np.transpose(state.board.array)).is_valid():
            return []

        result = state.adjacent_rule_actions()
        if result:
            return [result]
        result = state.sum_rule_actions()
        if result:
            return [result]
        result = state.get_first_empty_position_actions()
        return result

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        return state.fill(action)

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        return state.is_goal_state()

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        return node.state.get_h()


if __name__ == "__main__":
    # Ler o ficheiro de input de sys.argv[1],
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    board = Board.parse_instance_from_stdin()
    problem = Takuzu(board)
    goal_node = depth_first_tree_search(problem)
    print(goal_node.state.board)