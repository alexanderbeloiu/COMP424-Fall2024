# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
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

    def minimax(self, board, self_player, color, start, depth):
        legal_moves = get_valid_moves(board, color)
        max_score = float('-inf')
        min_score = float('inf')
        best_move = None
        max_depth = 2
        org_color = color
        if not self_player:
            if color == 1:
                color = 2
            else:
                color = 1

        
        if (depth > max_depth):
            _, player_score, opponent_score = check_endgame(board, org_color, 3 - org_color)
            return player_score - opponent_score
            #return player_score
        

        if (not legal_moves):
            print("NO LEGAL MOVES!!!!!!!\n\n\n-------------------------------------")
            return None
        
        for move in legal_moves:
            sim_board = deepcopy(board)

            if self_player:
                execute_move(sim_board, move, color)
                score = self.minimax(sim_board, False, color, False, depth + 1)
                if score > max_score:
                    max_score = score
                    best_move = move
            else:
                execute_move(sim_board, move, color)
                score = self.minimax(sim_board, True, color, False, depth + 1)
                if score < min_score:
                    min_score = score
                    best_move = move
            #move_score = self.evaluate_board(sim_board, player, player_score, opponent_score)

   
        if start:
            return best_move
        elif self_player:
            return max_score
        else:
            return min_score



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
        start_time = time.time()

        


        move = self.minimax(chess_board,True,color,True,0)





        time_taken = time.time() - start_time

        print("My AI's turn took ", time_taken, "seconds.")

        # Dummy return (you should replace this with your actual logic)
        # Returning a random valid move as an example
        return move #random_move(chess_board,player)

    
