from agents.agent import Agent
from store import register_agent
from helpers import get_valid_moves, execute_move, check_endgame
import copy
import random
import numpy as np

@register_agent("minimax_agent")
class StudentAgent(Agent):
    """
    A custom agent for playing Reversi/Othello using the minimax algorithm.
    """

    def __init__(self):
        super().__init__()
        self.name = "minimax_agent"
        self.color = None

    def step(self, board, color, opponent):
        """
        Choose a move using the minimax algorithm with alpha-beta pruning.

        Parameters:
        - board: 2D numpy array representing the game board.
        - color: Integer representing the agent's color (1 or 2).

        Returns:
        - Tuple (x, y): The coordinates of the chosen move.
        """
        if self.color is None:
            self.color = color  # Store the agent's color

        # Set the depth for the minimax algorithm
        depth = 3  # Adjust this value for performance vs. strength

        # Initialize alpha and beta for pruning
        alpha = float('-inf')
        beta = float('inf')

        # Get all legal moves for the current player
        legal_moves = get_valid_moves(board, color)

        if not legal_moves:
            return None  # No valid moves available, pass turn

        best_move = None
        best_value = float('-inf')

        # Evaluate each move using minimax
        for move in legal_moves:
            simulated_board = copy.deepcopy(board)
            execute_move(simulated_board, move, color)
            value = self.minimax(simulated_board, depth - 1, False, 3 - color, alpha, beta)
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # Alpha-beta pruning

        return best_move if best_move else random.choice(legal_moves)

    def minimax(self, board, depth, maximizingPlayer, color, alpha, beta):
        """
        Minimax algorithm with alpha-beta pruning.

        Parameters:
        - board: 2D numpy array representing the game board.
        - depth: Remaining depth to search.
        - maximizingPlayer: Boolean indicating if the current layer is maximizing.
        - color: Integer representing the current player's color.
        - alpha: Alpha value for pruning.
        - beta: Beta value for pruning.

        Returns:
        - float: The evaluated score of the board.
        """
        opponent_color = 3 - color
        legal_moves = get_valid_moves(board, color)
        game_over, _, _ = check_endgame(board, color, opponent_color)

        if depth == 0 or game_over:
            return self.evaluate_board(board, self.color)

        if not legal_moves:
            # No valid moves, pass turn to opponent
            return self.minimax(board, depth - 1, not maximizingPlayer, opponent_color, alpha, beta)

        if maximizingPlayer:
            max_eval = float('-inf')
            for move in legal_moves:
                new_board = copy.deepcopy(board)
                execute_move(new_board, move, color)
                eval = self.minimax(new_board, depth - 1, False, opponent_color, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                new_board = copy.deepcopy(board)
                execute_move(new_board, move, color)
                eval = self.minimax(new_board, depth - 1, True, opponent_color, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def evaluate_board(self, board, color):
        """
        Evaluate the board state based on multiple factors.

        Parameters:
        - board: 2D numpy array representing the game board.
        - color: Integer representing the agent's color.

        Returns:
        - float: The evaluated score of the board.
        """
        opponent_color = 3 - color

        # Basic evaluation: difference in number of pieces
        player_tiles = np.sum(board == color)
        opponent_tiles = np.sum(board == opponent_color)
        piece_diff = player_tiles - opponent_tiles

        # Corner positions are highly valuable
        corners = [
            (0, 0), (0, board.shape[1] - 1),
            (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)
        ]
        player_corners = sum([1 for corner in corners if board[corner] == color])
        opponent_corners = sum([1 for corner in corners if board[corner] == opponent_color])
        corner_diff = 25 * (player_corners - opponent_corners)

        # Mobility: difference in number of possible moves
        player_moves = len(get_valid_moves(board, color))
        opponent_moves = len(get_valid_moves(board, opponent_color))
        mobility = 0
        if player_moves + opponent_moves != 0:
            mobility = 10 * (player_moves - opponent_moves) / (player_moves + opponent_moves)

        # Combine the heuristics
        total_score = piece_diff + corner_diff + mobility
        #return total_score
        return piece_diff
