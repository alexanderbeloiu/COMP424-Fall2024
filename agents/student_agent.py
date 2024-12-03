# COMP 424 Final Project
# Alexander Beloiu

from agents.agent import Agent
from store import register_agent
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.real_color = None
        self.depths = []

    def sort_moves(self, board, color, moves):
        """
        Sort the moves based on the evaluation of the board after executing the move.
        
        Args:
            board (np.array): The current game board.
            color (int): The color of the agent.
            moves (list): List of possible moves.

        Returns:
            list: List of moves sorted based on the evaluation of the board.
        """
        scores = []

        # loop through moves and add score of executed move to list
        for move in moves:
            sim_board = deepcopy(board)
            execute_move(sim_board,move, color)
            score = self.evaluate_board(sim_board, self.real_color)
            scores.append(score)

        # sort by best moves for player
        if color == self.real_color:
            return [x for _, x in sorted(zip(scores, moves), key=lambda pair: pair[0], reverse=True)]

        # sort by worst moves that opponent makes given players perspective
        else:
            return [x for _, x in sorted(zip(scores, moves), key=lambda pair: pair[0], reverse=False)]

    def minimax(self, board, self_player, color, start, depth, max_depth, t, a=float('-inf'), b=float('inf')):
        """
        Minimax algorithm with alpha-beta pruning to determine the best move for the agent.
        
        Args:
            board (np.array): The current game board.
            self_player (bool): True if the agent is the max player.
            color (int): The color of the agent.
            start (bool): True if the function is at the root of the search tree.
            depth (int): The current depth of the search tree.
            max_depth (int): The maximum depth of the search tree.
            t (float): The time the search started.
            a (float): The alpha value for alpha-beta pruning.
            b (float): The beta value for alpha-beta pruning.

        Returns:
            float: The score of the best move if at the root of the search tree.
            tuple: The best move if not at the root of the search tree.
        """

        if (time.time() - t) > 1.95:
            # Timeout: Return a low score for max_player or high score for min_player
            return float('-inf') if self_player else float('inf')


        legal_moves = get_valid_moves(board, color)
        max_score = float('-inf')
        min_score = float('inf')
        best_move = None
        op_color = 3 - color
        
        fin, player_score, opponent_score = check_endgame(board, color, op_color)


        # If the game is over or the maximum depth is reached, return the board evaluation/score
        if (depth > max_depth) or fin:
            return self.evaluate_board(board, self.real_color)
        
        # If no moves are legal then pass the turn to the opponent.
        if not legal_moves:
            return self.minimax(deepcopy(board), not self_player, op_color, False, depth + 1, max_depth, t, a, b)


        #Move ordering
        legal_moves = self.sort_moves(board, color, legal_moves)

        # Loop through all possible moves
        for move in legal_moves:
            
            # Create copy of board for simulation.
            sim_board = deepcopy(board)

            # Execute move on the simulated board
            execute_move(sim_board, move, color)

            if self_player:

                # Evaluation of the board state with heuristics
                score = self.minimax(sim_board, False, 3-color, False, depth + 1, max_depth, t, a, b)

                # Update the score and best move
                if score >= max_score:
                    max_score = score
                    best_move = move
                
                # Alpha-beta pruning
                # If alpha > beta, prune the branch
                a = max(a, score)
                if (a >= b):
                    break
            else:
                score = self.minimax(sim_board, True, 3-color, False, depth + 1, max_depth, t, a, b)
                if score <= min_score:
                    min_score = score
                    best_move = move

                # Alpha-beta pruning
                # If alpha > beta, prune the branch
                b = min(b, score)
                if (a >= b):
                    break

        # Return the best move if at the root of the search
        if start:
            return best_move

        # Otherwise, return the best score
        return max_score if self_player else min_score



    def step(self, chess_board, color, opponent):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (board_size, board_size)
            where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
            and 2 represents Player 2's discs (Brown).
        - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
        - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

        You should return a tuple (r,c), where (r,c) is the position where your agent
        wants to place the next disc. Use functions in helpers to determine valid moves
        and more helpful tools.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # set the color of the agent
        if self.real_color is None:
            self.real_color = color

        legal_moves = get_valid_moves(chess_board, color)

        # attempt to stop freezing at the end
        # it doenst work....
        # maybe the freezing is normal
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]


        start_time = time.time()
        best_move = None
        mo = None
        depth = 1

        # Run Iterative Deepening Search
        # Stop when the time runs out. Set the best move to the move found in the last iteration
        # so we do not return a move of a partially evaluated minimax tree.
        while ((time.time() - start_time) < 1.95) and depth < 15:
            best_move = mo
            mo = self.minimax(deepcopy(chess_board),True,color,True,0,depth,start_time)
            depth += 1
    

        time_taken = time.time() - start_time

        print("My AI's turn took ", time_taken, "seconds.")

        return best_move

    
    def evaluate_board(self, board, color):
        """
        Evaluate the board state based on multiple factors.

        Args:
            board: 2D numpy array representing the game board.
            color: Integer representing the agent's color.

        Returns:
            float: The evaluated score of the board.
        """
        opponent_color = 3 - color

        # Basic evaluation: difference in number of pieces
        player_tiles = np.sum(board == color)
        opponent_tiles = np.sum(board == opponent_color)
        piece_diff = player_tiles - opponent_tiles

        # Corner positions are valuable
        # Calculaute the difference in number of corners occupied by the player and the opponent
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
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
        return total_score
