# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves
import random
import math


@register_agent("student_agent_prev")
class StudentAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgentPrev"
        self.real_color = None
        self.depths = []


    def minimax(self, board, self_player, color, start, depth, max_depth, t, a=float('-inf'), b=float('inf')):
        if (time.time() - t) > 1.9:
            # Timeout: Return a very low score for max_player or high score for min_player
            return float('-inf') if self_player else float('inf')

        legal_moves = get_valid_moves(board, color)
        #MAKE IT SO THAT EDGE PIECES ARE PRIORITIZED?
        random.shuffle(legal_moves)
        max_score = float('-inf')
        min_score = float('inf')
        best_move = None
        op_color = 3 - color
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        
        fin, player_score, opponent_score = check_endgame(board, color, op_color)
        if self.real_color is None:
            self.real_color = color


        if (depth > max_depth) or fin:
            _, player_score, opponent_score = check_endgame(board, self.real_color, 3-self.real_color)
            self.depths.append(depth)
            return self.evaluate_board(board, self.real_color)
        

        if not legal_moves:
            # If no moves are legal, evaluate the board or return a neutral score
            #return player_score - opponent_score
            return self.minimax(deepcopy(board), not self_player, op_color, False, depth + 1, max_depth, t, a, b)


        #print("\n---------------------------\n Depth: ", depth, "Breadth: ", len(legal_moves))
        # Loop through all possible moves
        i=1
        for move in legal_moves:
            sim_board = deepcopy(board)
            # if move in corners:
            #     max_score = 100
            #     best_move = move
            #     break

            if self_player:
                execute_move(sim_board, move, color)
                score = self.minimax(sim_board, False, 3-color, False, depth + 1, max_depth, t, a, b)
                if score >= max_score:
                    max_score = score
                    best_move = move
                a = max(a, score)
                if (a >= b) and depth > 1:
                    #self.depths.append(1-(i/len(legal_moves)))
                    break
            else:
                execute_move(sim_board, move, color)
                score = self.minimax(sim_board, True, 3-color, False, depth + 1, max_depth, t, a, b)
                if score <= min_score:
                    min_score = score
                    best_move = move
                b = min(b, score)
                if (a >= b) and depth > 1:
                    #self.depths.append(1-(i/len(legal_moves)))
                    break
            i+=1
        # Return the best move if at the root of the search
        if start:
            if best_move is None:
                print("BEST MOVE IS NONE!")
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

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.

        legal_moves = get_valid_moves(chess_board, color)

        #attempt to stop freezing at the end
        #it doenst work....
        #maybe the freezing is normal
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]



        start_time = time.time()

        best_move = None
        mo = None
        depth = 1
        while (time.time() - start_time) < 1.9:
            best_move = mo
            #mo = self.minimax(deepcopy(chess_board),False,3-color,True,0,depth,start_time)
            mo = self.minimax(deepcopy(chess_board),True,color,True,0,depth,start_time)
            depth += 1
        print("\n---------------------------\n2Final Depth: ", depth)

        print("2Mean Depth: ", sum(self.depths)/len(self.depths))
        print("2Max Depth: ", max(self.depths))
        print("2Min Depth: ", min(self.depths))
        print("2median Depth: ", sorted(self.depths)[len(self.depths)//2])
        print("2Number of Depths: ", len(self.depths))
        print("2Mode Depth: ", max(set(self.depths), key = self.depths.count))

        time_taken = time.time() - start_time

        print("2My AI's turn took ", time_taken, "seconds.")

        # Dummy return (you should replace this with your actual logic)
        # Returning a random valid move as an example
        return best_move #random_move(chess_board,player)

    
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
        return total_score
        #return piece_diff
