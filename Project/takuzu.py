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

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        return (self.get_number(row, col-1) if col > 0 else None, self.get_number(row, col+1) if col < (self.get_dimension_len(1)-1) else None)

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
            line_array = [np.fromstring(line[:-1], int, sep='\t')]
            board = np.append(board, line_array, axis=0)
        
        return Board(board)

    # TODO: outros metodos da classe

    def resulting_board(self, row: int, col: int, val: int):
        new_array = np.copy(self.array)
        new_array[row, col] = val

        return Board(new_array)

    def __repr__(self) -> str:
        return '\n'.join(['\t'.join(['{}'.format(number) for number in row]) for row in self.array])
    

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
        return self.check_conditions(self.board) and self.check_conditions(Board(np.transpose(self.board.array)))

    def check_conditions(self, board: Board) -> bool:
        dimension_zero = board.get_dimension_len(0)
        dimension_one = board.get_dimension_len(1)
        for i in range(dimension_zero):
            sumLine = 0
            for j in range(dimension_one):
                num = board.get_number(i, j)
                if num == 2:
                    return False
                sumLine += num
                if (j > 0 and j < dimension_one-1):
                    adjacent = board.adjacent_horizontal_numbers(i, j)
                    if (num == adjacent[0] and num == adjacent[1]):
                        return False
            if (dimension_one % 2 == 0):
                if (sumLine != dimension_one//2):
                    return False
            else:
                if (sumLine != (dimension_one//2)+1 or sumLine != (dimension_one//2)):
                    return False
            
            for n in range(i+1, dimension_zero):
                if np.array_equal(board.array[i], board.array[n]):
                    return False
        return True
       

class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = TakuzuState(board)

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        pass

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
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro de input de sys.argv[1],
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass


board = Board.parse_instance_from_stdin()
problem = Takuzu(board)
s0 = TakuzuState(board)
print("Initial:\n", s0.board, sep="")

s1 = problem.result(s0, (0, 0, 0))
s2 = problem.result(s1, (0, 2, 1))
s3 = problem.result(s2, (1, 0, 1))
s4 = problem.result(s3, (1, 1, 0))
s5 = problem.result(s4, (1, 3, 1))
s6 = problem.result(s5, (2, 0, 0))
s7 = problem.result(s6, (2, 2, 1))
s8 = problem.result(s7, (2, 3, 1))
s9 = problem.result(s8, (3, 2, 0))

print("Is goal?", problem.goal_test(s9))
print("Solution:\n", s9.board, sep="")


# board1 = board.resulting_board(0, 0, 0)
# print("Old Board:\n", board, sep="")
# print("New Board:\n", result_state.board, sep="")
# print(board.adjacent_horizontal_numbers(2, 2))
# print(board.adjacent_vertical_numbers(2, 2))