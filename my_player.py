"""
votre agent. Vous devez modifier ce fichier.
"""

from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import math
import time
import random
import numpy as np


class MyPlayer(PlayerAbalone):
    """
    Player class for Abalone game.

    Attributes:
        piece_type (str): piece type of the player
    """

    log = True #True if you want to write the log of each action in a text file

    #Coordinate system conversion into cube coordinates
    #https://www.redblobgames.com/grids/hexagons/#coordinates-cube
    to_cube = { (4,0) : (0,4,-4), (3,1) : (1,3,-4), (2,2) : (2,2,-4), (1,3) : (3,1,-4), (0,4) : (4,0,-4),
                (6,0) : (-1, 4,-3), (5,1) : (0,3,-3), (4,2) : (1,2,-3), (3,3) : (2,1,-3), (2,4) : (3,0,-3), (1,5) : (4,-1,-3),
                (8,0) : (-2,4,-2), (7,1) : (-1,3,-2), (6,2) : (0,2,-2), (5,3) : (1,1,-2), (4,4) : (2,0,-2), (3,5) : (3,-1,-2), (2,6) : (4,-2,-2),
                (10,0) : (-3,4,-1), (9,1) : (-2,3,-1), (8,2) : (-1,2,-1), (7,3) : (0,1,-1), (6,4) : (1,0,-1), (5,5) : (2,-1,-1), (4,6) : (3,-2,-1), (3,7) : (4,-3,-1),
                (12,0) : (-4,4,0), (11,1) : (-3,3,0), (10,2) : (-2,2,0), (9,3) : (-1,1,0), (8,4) : (0,0,0), (7,5) : (1,-1,0), (6,6) : (2,-2,0), (5,7) : (3,-3,0), (4,8) : (4,-4,0),
                (13,1) : (-4,3,1), (12,2) : (-3,2,1), (11,3) : (-2,1,1), (10,4) : (-1,0,1), (9,5) : (0,-1,1), (8,6) : (1,-2,1), (7,7) : (2,-3,1), (6,8) : (3,-4,1),
                (14,2) : (-4,2,2), (13,3) : (-3,1,2), (12,4) : (-2,0,2), (11,5) : (-1,-1,2), (10,6) : (0,-2,2), (9,7) : (1,-3,2), (8,8) : (2,-4,2), 
                (15,3) : (-4,1,3), (14,4) : (-3,0,3), (13,5) : (-2,-1,3), (12,6) : (-1,-2,3), (11,7) : (0,-3,3), (10,8) : (1,-4,3),
                (16,4) : (-4,0,4), (15,5) : (-3,-1,4), (14,6) : (-2,-2,4), (13,7) : (-1,-3,4), (12,8) : (0,-4,4)
                }

    to_coord = { (0,4,-4) : (4,0), (1,3,-4) : (3,1), (2,2,-4) : (2,2), (3,1,-4) : (1,3), (4,0,-4) : (0,4),
                (-1, 4,-3) : (6,0), (0,3,-3) : (5,1), (1,2,-3) : (4,2), (2,1,-3) : (3,3), (3,0,-3) : (2,4), (4,-1,-3) : (1,5),
                (-2,4,-2) : (8,0), (-1,3,-2) : (7,1), (0,2,-2) : (6,2), (1,1,-2) : (5,3), (2,0,-2) : (4,4), (3,-1,-2) : (3,5), (4,-2,-2) : (2,6),
                (-3,4,-1) : (10,0), (-2,3,-1) : (9,1), (-1,2,-1) : (8,2), (0,1,-1) : (7,3), (1,0,-1) : (6,4), (2,-1,-1) : (5,5), (3,-2,-1) : (4,6), (4,-3,-1) : (3,7),
                (-4,4,0) : (12,0), (-3,3,0) : (11,1), (-2,2,0) : (10,2), (-1,1,0) : (9,3), (0,0,0) : (8,4), (1,-1,0) : (7,5), (2,-2,0) : (6,6), (3,-3,0) : (5,7), (4,-4,0) : (4,8),
                (-4,3,1) : (13,1), (-3,2,1) : (12,2), (-2,1,1) : (11,3), (-1,0,1) : (10,4), (0,-1,1) : (9,5), (1,-2,1) : (8,6), (2,-3,1) : (7,7), (3,-4,1) : (6,8),
                (-4,2,2) : (14,2), (-3,1,2) : (13,3), (-2,0,2) : (12,4), (-1,-1,2) : (11,5), (0,-2,2) : (10,6), (1,-3,2) : (9,7), (2,-4,2) : (8,8), 
                (-4,1,3) : (15,3), (-3,0,3) : (14,4), (-2,-1,3) : (13,5), (-1,-2,3) : (12,6), (0,-3,3) : (11,7), (1,-4,3) : (10,8),
                (-4,0,4) : (16,4), (-3,-1,4) : (15,5), (-2,-2,4) : (14,6), (-1,-3,4) : (13,7), (0,-4,4) : (12,8)
                }

    grid_to_coord = { (0,6) : (0,4), (0,5) : (1,3), (0,4) : (2,2), (0,3) : (3,1), (0,2) : (4,0),
                        (1,6) : (1,5), (1,5) : (2,4), (1,4) : (3,3), (1,3) : (4,2), (1,2) : (5,1), (1,1) : (6,0),
                        (2,7) : (2,6), (2,6) : (3,5), (2,5) : (4,4), (2,4) : (5,3), (2,3) : (6,2), (2,2) : (7,1), (2,1) : (8,0),
                        (3,7) : (3,7), (3,6) : (4,6), (3,5) : (5,5), (3,4) : (6,4), (3,3) : (7,3), (3,2) : (8,2),
                        (3,1) : (9,1), (4,8) : (4,8), (4,7) : (5,7), (4,6) : (6,6), (4,5) : (7,5), (4,4) : (8,4),
                        (4,3) : (9,3), (5,7) : (6,8), (5,6) : (7,7), (5,5) : (8,6), (5,4) : (9,5), (6,7) : (8,8),
                        (6,6) : (9,7), (5,3) : (10,4), (5,2) : (11,3), (5,1) : (12,2), (5,0) : (13,1), (4,2) : (10,2),
                        (4,1) : (11,1), (4,0) : (12,0), (6,5) : (10,6), (6,4) : (11,5), (6,3) : (12,4), (6,2) : (13,3),
                        (6,1) : (14,2), (7,6) : (10,8), (7,5) : (11,7), (7,4) : (12,6), (7,3) : (13,5), (7,2) : (14,4),
                        (7,1) : (15,3), (8,6) : (12,8), (8,5) : (13,7), (8,4) : (14,6), (8,3) : (15,5), (8,2) : (16,4),
                        (3,0) : (10,0)
                }
                


    def __init__(self, piece_type: str, name: str = "bob", time_limit: float=60*15,*args) -> None:
        """
        Initialize the PlayerAbalone instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type,name,time_limit,*args)

        #transposition table to store the visited nodes
        self.transposition_table = {}
        self.zobrist_table = [[[[random.randint(1,2**64 - 1) for i in range(2)] for j in range(8)] for k in range(8)] for l in range(8)]

        self.visited_nodes = 0
        self.cumulated_time = 0
        self.action_number = 0
        self.text_file_log = f"log_{self.piece_type}.txt"
        if MyPlayer.log:
            #Initialize the log file
            with open(self.text_file_log, "w") as f:
                f.write("")


    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Function to implement the logic of the player.

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """
        self.time_consumed = time.time()

        maximizing_player = self.piece_type == "W"
        depth = 3

        value, best_action = self.minmax_alphabeta(current_state, depth, -math.inf, math.inf, maximizing_player)

        self.time_consumed = time.time() - self.time_consumed
        self.cumulated_time += self.time_consumed

        self.action_number += 1
        if MyPlayer.log:
            with open(self.text_file_log, "a") as f:
                #f.write(f"Zobrist : {self.compute_zobristhash(current_state)}, ")
                f.write(f"Action {self.action_number} : {self.time_consumed} s (cumulated time {self.cumulated_time} s), {self.visited_nodes} visited nodes, {value} value\n")
                
        self.visited_nodes = 0

        return best_action

    
    def compute_zobristhash(self, current_state: GameState): 
        """
        Function to compute the Zobrist Hashing of the current state.

        https://iq.opengenus.org/zobrist-hashing-game-theory/
        """
        h = 0

        #Get the environment
        for coord, player in current_state.get_rep().get_env().items() :
            x,z,y = MyPlayer.to_cube[coord]
            h ^= self.zobrist_table[x][z][y][int(player.get_type() == "W")]

        return h

    
    def winner_score(self, current_state: GameState) -> float:
        """
        Function to implement the winner score when the game is finished.
        """

        #Get player list
        players = current_state.get_players()
        player_score = current_state.get_scores()
        assert len(players) == 2

        w_id = players[0].get_id() if players[0].get_piece_type() == "W" else players[1].get_id()
        b_id = players[0].get_id() if players[0].get_piece_type() == "B" else players[1].get_id()

        if player_score[b_id] == -6 : #White player won
            return 100
        elif player_score[w_id] == -6 : #Black player won
            return -100
        else :
            return 2*(player_score[w_id] - player_score[b_id])

    
    def utility(self, current_state: GameState, depth: int, maximizing_player : bool) -> float:
        """
        Function to implement the utility function.

        Args:
            current_state (GameState): Current game state representation
            depth (int): depth of the tree
            maximizing_player (bool): True if the player is maximizing

        Returns:
            float: the value of the utility function
        """
        if current_state.is_done():
            return self.winner_score(current_state)
        elif depth == 0:
            return self.heuristic(current_state)
        else :
            raise Exception("Utility used on a non terminal node")

    
    def heuristic(self, current_state: GameState) -> float:
        """
        Function to implement the heuristic function to evaluate the value of a node.

        1) Number of marbles difference (max 5 because if 6, a player has necessarily won)
        2) Number of marbles on the edge
        3) Sum of Manhattan distance to the center
        4) Marbles cohesion

        Args:
            current_state (GameState): Current game state representation

        Returns:
            float: the value of the heuristic function
        """

        marble_count = {"W" : 0, "B" : 0}
        distance_center = {"W" : 0, "B" : 0}
        edge_count = {"W" : 0, "B" : 0}
        cohesion = {"W" : 0, "B" : 0}

        #Loop over the environnement current_state.get_rep().get_env() = {  (a,b) :  <seahorse.game.game_layout.board.Piece> , ... }
        for coord, marble in current_state.get_rep().get_env().items() :
            color_marble = marble.get_type()
            x,z,y = MyPlayer.to_cube[coord]

            #Update the marble count
            marble_count[color_marble] += 1

            #Update the distance to the center that is actually adding the max absolute value of the cube coordinates
            maxi_cube_coord = max(abs(x), abs(y), abs(z))
            distance_center[color_marble] += maxi_cube_coord
            
            #Update the edge count
            if maxi_cube_coord == 4:
                edge_count[color_marble] += 1
            
            #Update the cohesion (number of neighbours of the same color)
            neighbours_cube_coord = [  (x+1, z-1, y), (x+1, z, y-1), (x, z+1, y-1), (x-1, z+1, y), (x-1, z, y+1), (x, z-1, y+1) ]
            for x_n, y_n, z_n in neighbours_cube_coord:
                neighbour_coord = MyPlayer.to_coord.get( (x_n, y_n, z_n) , -1) 
                neighbour = current_state.get_rep().find( neighbour_coord )
                if neighbour != -1 and neighbour.get_type() == color_marble:
                    cohesion[color_marble] += 1
            
        #Compute the heuristic value
        heuristic_value = 0

        #1) Number of marbles difference (max 5 because if 6, a player has necessarily won)
        heuristic_value += marble_count["W"] - marble_count["B"]

        #2) Number of marbles on the edge
        heuristic_value += edge_count["B"]/marble_count["B"] - edge_count["W"]/marble_count["W"]

        #3) Sum of Manhattan distance to the center
        heuristic_value += (distance_center["B"]/marble_count["B"] - distance_center["W"]/marble_count["W"])/4

        #4) Marbles cohesion
        heuristic_value += (cohesion["B"]/marble_count["B"] - cohesion["W"]/marble_count["W"])/6

        return heuristic_value


    def guess_value(self, action : Action) :
        """
        Function to guess the value of an action (for ordering the actions).
        """
        current_env = action.get_current_game_state().get_rep().get_env()
        next_env = action.get_next_game_state().get_rep().get_env()

        push = False
        get_out_of_edge = False


        for coord, piece in current_env.items():
            next_piece =  next_env.get(coord, -1)

            x,y,z = MyPlayer.to_cube[coord]
            piece_type = piece.get_type()
            piece_type_next = next_piece.get_type() if next_piece != -1 else -1

            #Push occured if a marble replaced another one of the opposite color
            if (piece_type == "W" and piece_type_next == "B") or (piece_type == "B" and piece_type_next == "W") :
                push = True
            
            #See if a marble in a edge has moved
            if max(abs(x), abs(y), abs(z)) == 4 :
                if piece_type != piece_type_next :
                    get_out_of_edge = True

            #Stop the loop
            if push and get_out_of_edge :
                break

        return int(push) + int(get_out_of_edge)

        """
        #Some transformations to get the difference between the two grids
        current_grid = current_state.get_rep().get_grid()
        current_grid = np.array(   [[ 1 if current_grid[i][j] == "W" else -1 if current_grid[i][j] == "B" else 0  for j in range(9)   ]   for i in range(9)  ]   )
        next_grid = next_state.get_rep().get_grid()
        next_grid = np.array(   [[ 1 if next_grid[i][j] == "W" else -1 if next_grid[i][j] == "B" else 0  for j in range(9)   ]   for i in range(9)  ]   )
        diff_grid = next_grid - current_grid

        # Check if the action is a push : if we find a 2 or a -2 in the difference grid
        push =  2 in diff_grid or -2 in diff_grid

        #Count the number of marbles in edge
        get_out_of_edge = False
        player_is_white = self.get_piece_type() == "W"
        for grid_coord, coord in MyPlayer.grid_to_coord.items():
            x,y,z = MyPlayer.to_cube[coord] #Get the cube coordinates
            if next_grid[grid_coord] == int(player_is_white) and max(abs(x), abs(y), abs(z)) == 4 :
                get_out_of_edge = True
                break
    
        return int(push) + int(get_out_of_edge)
        """


    def ordering_actions(self, current_state: GameState, possible_actions : list[Action]) -> list[Action]:
        """
        Function to order the actions.

        Args:
            current_state (GameState): Current game state representation
            possible_actions (list[Action]): list of possible actions

        Returns:
            list[Action]: the ordered list of actions
        """

        action_guess = { action : self.guess_value(action) for action in possible_actions }

        #Get the list of actions sorted by the guess value
        return sorted(action_guess, key=action_guess.get, reverse=True)



    def minmax_alphabeta(self, current_state: GameState, depth: int, alpha : float, beta : float, maximizing_player : bool) -> float:
        """
        Function to implement the minmax algorithm with alpha beta pruning.

        Args:
            current_state (GameState): 
            depth (int): depth of the tree
            alpha (float): alpha value
            beta (float): beta value
            maximizing_player (bool): True if the player is maximizing

        Returns:
            float: the value of the node
        """
        
        #Terminal node
        if depth == 0 or current_state.is_done():
            return self.utility(current_state, depth, maximizing_player), None

        """historic_log/
        #Check in the transposition table
        #Transposition table has the form : { zobrist_hash : (depth, value, action) }
        state_hash = self.compute_zobristhash(current_state)
        if state_hash in self.transposition_table :
            if self.transposition_table[state_hash][0] >= depth :
                return self.transposition_table[state_hash][1], self.transposition_table[state_hash][2]
        """
    


        #Initialize score and action
        best_score, best_action = -math.inf if maximizing_player else math.inf, None


        #Find the best successor
        possible_actions = self.ordering_actions( current_state, current_state.get_possible_actions() )
        #possible_actions = current_state.get_possible_actions()
        for action in possible_actions:

            next_state = action.get_next_game_state()
            score, _ = self.minmax_alphabeta(next_state, depth-1, alpha, beta, not maximizing_player)
            self.visited_nodes += 1

            if maximizing_player:
                if score > best_score:
                    best_score, best_action = score, action
                alpha = max(alpha, best_score)
                if best_score >= beta:
                    return best_score, best_action
            else:
                if score < best_score:
                    best_score, best_action = score, action
                beta = min(beta, best_score)
                if best_score <= alpha:
                    return best_score, best_action
            

        return best_score, best_action




            
